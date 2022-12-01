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


"""post process for 310 inference"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from src.data.coco_eval import CocoEvaluator


def prepare_args():
    parser = argparse.ArgumentParser(description="postprocess")
    parser.add_argument("--result_dir", type=str, default="./result_Files", help="result files path.")
    parser.add_argument('--anno_path', type=str)
    args = parser.parse_args()
    return args


def softmax(x):
    """ softmax function """
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes[:, :1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
    x0, y0, x1, y1 = x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h
    return np.concatenate([x0, y0, x1, y1], axis=1)


def call_map(args):
    coco_gt = COCO(args.anno_path)
    img_ids = coco_gt.getImgIds()
    coco_evaluator = CocoEvaluator(coco_gt, ('bbox',))

    print("\n========================================\n")
    print("Processing, please wait a moment.")

    for img_id in tqdm(img_ids):
        file_id = str(img_id).zfill(12)

        label_result_file = os.path.join(args.result_dir, file_id + "_0.bin")
        bbox_result_file = os.path.join(args.result_dir, file_id + "_1.bin")

        out_logits = np.fromfile(label_result_file, dtype=np.float16).reshape(-1, 92)
        out_bbox = np.fromfile(bbox_result_file, dtype=np.float16).reshape(-1, 4)

        prob = softmax(out_logits)

        labels = np.argmax(prob[..., :-1], axis=-1)
        bs_idx = np.arange(prob.shape[0])
        scores = prob[bs_idx, labels]

        img_h = coco_gt.loadImgs(img_id)[0]['height']
        img_w = coco_gt.loadImgs(img_id)[0]['width']
        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = np.array([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct

        results = {'scores': scores, 'labels': labels, 'boxes': boxes}
        res = {img_id: results}
        coco_evaluator.post_process_update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    print("\n========================================\n")


if __name__ == '__main__':
    args = prepare_args()
    call_map(args)
