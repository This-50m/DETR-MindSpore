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

import os

import cv2
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from src import prepare_args
from src.data.dataset import coco_id_dict
from src.DETR.util import box_cxcywh_to_xyxy
from src.DETR.backbone import build_backbone
from src.DETR.detr import build_transformer, DETR


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    h, w = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)


class SOTAResize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        h, w = img.shape[:2]
        nh, nw = get_size_with_aspect_ratio((h, w), self.min_size, self.max_size)
        size = (nh, nw)
        resize_img = cv2.resize(img, (nw, nh))
        print(f'resize size: {nh},{nw}')
        mask = np.zeros((nh, nw), dtype=np.bool_)
        return resize_img, mask, size


class SOTAPad(object):
    def __init__(self, tgt_size):
        self.tgt_size = tgt_size

    def __call__(self, img, mask):
        c, h, w = img.shape
        new_img = np.zeros((c, self.tgt_size, self.tgt_size), dtype=np.float32)
        new_img[:, :h, :w] = img
        new_mask = np.ones((self.tgt_size, self.tgt_size), dtype=np.float32)
        new_mask[:h, :w] = 0
        return new_img, new_mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)
        return image


def save_img(image, name):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.uint8)
    name = name + '.jpg'
    cv2.imwrite(name, image)


def save_result(image, scores, bboxes, labels):
    scores = np.squeeze(scores.asnumpy())
    bboxes = np.squeeze(bboxes.asnumpy())
    labels = np.squeeze(labels.asnumpy())

    for score, bbox, label in zip(scores, bboxes, labels):
        if score > 0.9:
        # if score > 0.003:
            x0, y0, x1, y1 = list(map(int, bbox))
            print(f'[{x0},{y0},{x1},{y1}]==[{coco_id_dict[label]}]==[{score}]')
            image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
            image = cv2.putText(image, coco_id_dict[label], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    save_img(image, 'result_img')


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


def detect_image():
    # config
    args = prepare_args()
    args.aux_loss = False

    # context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    # build model and load checkpoint
    net = build_net(args)
    net.set_train(False)
    ckpt = load_checkpoint('ms_detr_sota.ckpt')
    new_ckpt = {}
    for k, v in ckpt.items():
        if 'optimizer.' in k:
            k = k.replace('optimizer.', '')
        if 'network.net.' in k:
            k = k.replace('network.net.', '')
        new_ckpt[k] = v
    load_param_into_net(net, new_ckpt, strict_load=True)

    # load image
    img = cv2.imread('demo.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_img = img

    # transform
    h, w, _ = img.shape
    print(f'ori size: {h},{w}')

    trans1 = SOTAResize(800, 1333)
    trans2 = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    trans3 = SOTAPad(1344)

    img, mask, new_size = trans1(img)
    img = trans2(img)
    img, mask = trans3(img, mask)
    re_h, re_w = new_size

    # build model input tensor_list
    tensor = ms.Tensor(img[None], dtype=ms.float32)
    mask = ms.Tensor(mask[None], dtype=ms.float32)

    # forward
    print(tensor.shape)
    print(mask.shape)
    r = net(tensor, mask)
    pred_logits, pred_boxes = r
    print(pred_logits.shape)
    print(pred_boxes.shape)

    # post process
    prob = nn.Softmax()(pred_logits)
    labels, scores = ops.ArgMaxWithValue(axis=-1)(prob[..., :-1])
    boxes = box_cxcywh_to_xyxy(pred_boxes)
    scale_fct = ms.Tensor([w, h, w, h], dtype=ms.float32)
    boxes = boxes * scale_fct[None, None, :]

    # save result
    save_result(ori_img, scores, boxes, labels)


if __name__ == '__main__':
    detect_image()
