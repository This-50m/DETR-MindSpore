"""
DETR model and criterion classes.
"""
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

from mindspore import nn
from mindspore import ops
from mindspore import ms_function
from mindspore.common import initializer as init

from src.DETR.init_weights import KaimingUniform, UniformBias
from src.DETR.util import box_cxcywh_to_xyxy
from src.DETR.backbone import build_backbone
from src.DETR.transformer import build_transformer
from src.DETR.criterion import SetCriterion


class MLP(nn.Cell):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        input_channel = [input_dim] + [hidden_dim] * (num_layers-1)
        output_channel = [hidden_dim] * (num_layers-1) + [output_dim]
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Dense(input_channel[i], output_channel[i], weight_init=KaimingUniform()))
            layers.append(nn.ReLU())
        layers.append(nn.Dense(input_channel[-1], output_channel[-1]))
        self.layers = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.layers(x)


class DETR(nn.Cell):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=True):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Dense(hidden_dim, num_classes + 1,
                                    weight_init=KaimingUniform(),
                                    bias_init=UniformBias([num_classes + 1, hidden_dim]))
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim, embedding_table=init.Normal(sigma=1.0))
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1, has_bias=True,
                                    weight_init=KaimingUniform(),
                                    bias_init=UniformBias([num_classes + 1, hidden_dim]))
        self.backbone = backbone
        self.sigmoid = nn.Sigmoid()
        self.aux_loss = aux_loss
        self.cast = ops.Cast()

    @ms_function
    def construct(self, x, mask):
        """
            tensor: batched images, of shape [batch_size x 3 x H x W]
            mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        src, mask, pos = self.backbone(x, mask)

        query_embed = self.query_embed.embedding_table
        src = self.input_proj(src)

        # adaptive float16 or float32
        query_embed = self.cast(query_embed, src.dtype)
        pos = self.cast(pos, src.dtype)

        hs = self.transformer(src, mask, query_embed, pos)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)
        outputs_coord = self.sigmoid(outputs_coord)

        if not self.aux_loss:
            # (bs, h, w)
            pred_logits = outputs_class[-1]
            pred_boxes = outputs_coord[-1]
        else:
            # (head, bs, h, w)
            pred_logits = outputs_class
            pred_boxes = outputs_coord
        return pred_logits, pred_boxes


class PostProcess(object):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self):
        super(PostProcess, self).__init__()
        self.softmax = nn.Softmax()
        self.argmax = ops.ArgMaxWithValue(axis=-1)
        self.unstack = ops.Unstack(axis=1)
        self.stack = ops.Stack(axis=1)

    def __call__(self, out_logits, out_bbox, target_sizes):
        """ Perform the computation
        Parameters:
            out_logits, out_bbox: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        prob = self.softmax(out_logits)
        # exclude last one
        labels, scores = self.argmax(prob[..., :-1])

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = self.unstack(target_sizes)
        scale_fct = self.stack([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def build(args):
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
    if args.device_target == 'Ascend' and args.context_mode == "GRAPH":
        from src.DETR.matcher import build_matcher
        matcher = build_matcher(args)
    else:
        from src.DETR.matcher_np import build_matcher
        matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    criterion = SetCriterion(args,
                             num_classes=num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             aux_loss=args.aux_loss)

    postprocessors = PostProcess()

    return model, criterion, postprocessors
