# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import time
import os
from collections import OrderedDict
from tqdm import tqdm

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from pycocotools.coco import COCO

from src import prepare_args
from src.data.coco_eval import CocoEvaluator
from src.DETR.util import box_cxcywh_to_xyxy
from src.data.dataset import create_mindrecord, create_detr_dataset
from src.DETR.backbone import build_backbone
from src.DETR.detr import build_transformer, DETR


def load_ckpt(weights_path):
    ckpt = load_checkpoint(weights_path)
    new_ckpt = {}
    for k, v in ckpt.items():
        if 'optimizer.' in k:
            k = k.replace('optimizer.', '')
        if 'network.net.' in k:
            k = k.replace('network.net.', '')
        new_ckpt[k] = v
    return new_ckpt


def build_net(args):
    num_classes = args.num_classes

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )
    return model


def evaluation():
    args = prepare_args()
    args.aux_loss = False

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(device_id=args.device_id)

    net = build_net(args)
    net.set_train(False)

    load_param_into_net(net, load_ckpt(args.resume), strict_load=True)

    net.to_float(ms.float16)
    if args.device_target == "GPU":
        for _, cell in net.cells_and_names():
            if isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm)):
                cell.to_float(ms.float32)

    mindrecord_file = create_mindrecord(args, 0, "DETR.mindrecord.eval", False)
    ds = create_detr_dataset(args, mindrecord_file, batch_size=16,
                             device_num=1, rank_id=0,
                             num_parallel_workers=args.num_parallel_workers,
                             python_multiprocessing=args.python_multiprocessing,
                             is_training=False)
    total = ds.get_dataset_size()

    anno_json = os.path.join(args.coco_path, "annotations/instances_{}.json".format(args.val_data_type))
    coco_gt = COCO(anno_json)
    coco_evaluator = CocoEvaluator(coco_gt, ('bbox', ))

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    start = time.time()
    results = []
    for data in tqdm(ds.create_dict_iterator(output_numpy=True)):
        # image, mask, image_id, ori_size = data
        image = Tensor(data['image'], ms.float16)
        mask = Tensor(data['mask'], ms.float16)
        ori_size = Tensor(data['ori_size'])
        image_id = data['image_id']

        out_logits, out_bbox = net(image, mask)

        prob = ops.Softmax()(out_logits)
        labels, scores = ops.ArgMaxWithValue(axis=-1)(prob[..., :-1])
        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = ops.Unstack(axis=1)(ori_size)
        scale_fct = ops.Stack(axis=1)([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct[:, None, :]

        results.append((image_id, (scores, labels, boxes)))
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        # res = {idx: output for idx, output in zip(image_id, results)}
        # coco_evaluator.update(res)
        # coco_evaluator.graph_update(image_id, scores, labels, boxes)

    for image_id, (scores, labels, boxes) in results:
        res = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        img_res = {idx: output for idx, output in zip(image_id, res)}
        coco_evaluator.update(img_res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    print(coco_evaluator.coco_eval.get('bbox').stats)
    print('cost time: ', time.time() - start)
    print("\n========================================\n")


if __name__ == '__main__':
    evaluation()