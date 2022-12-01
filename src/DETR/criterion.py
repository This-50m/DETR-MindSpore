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
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore import ms_function
from src.DETR.util import box_cxcywh_to_xyxy, generalized_box_iou


class LogSoftmaxCrossEntropyWithLogits(nn.Cell):
    def __init__(self, weights):
        super(LogSoftmaxCrossEntropyWithLogits, self).__init__()
        self.log_soft_max = nn.LogSoftmax()
        self.nll_loss = ops.NLLLoss(reduction='mean')
        self.weights = Tensor(weights, dtype=mstype.float32)
        self.reshape = ops.Reshape()

    @ms_function
    def construct(self, logits, labels):
        """
        :param logits: (Bs, N, classes). float32
        :param labels: (Bs, N), int32
        :return: loss
        """
        bs, n, cls = logits.shape
        logits = self.reshape(logits, (bs * n, cls))
        labels = self.reshape(labels, (bs * n,))

        logits = self.log_soft_max(logits)
        loss, _ = self.nll_loss(logits, labels, self.weights)
        return loss


class BoxLoss(nn.Cell):
    def __init__(self):
        super(BoxLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.div = ops.Div()
        self.sub = ops.Sub()
        self.reduce_sum = ops.ReduceSum()
        self.reshape = ops.Reshape()

    @ms_function
    def construct(self, pred_boxes, target_boxes, boxes_valid):
        """
        :param pred_boxes: (Bs, Queries, 4). float32
        :param target_boxes: (Bs, Queries, 4). float32
        :param boxes_valid: (Bs, Queries). float32
        :return:
        """
        bs, query, _ = pred_boxes.shape
        num_boxes = self.reduce_sum(boxes_valid)

        # reshape (bs*query, 4)
        pred_boxes = self.reshape(pred_boxes, (bs * query, 4))
        target_boxes = self.reshape(target_boxes, (bs * query, 4))

        # compute l1 loss
        loss_bbox = self.l1_loss(pred_boxes, target_boxes)
        loss_bbox = loss_bbox * self.reshape(boxes_valid, (bs * query, 1))
        loss_bbox = self.reduce_sum(loss_bbox) / num_boxes

        # compute giou loss
        giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))
        loss_giou = self.sub(1, giou.diagonal()) * self.reshape(boxes_valid, (bs * query,))
        loss_giou = self.reduce_sum(loss_giou) / num_boxes

        return loss_bbox, loss_giou


class SetCriterion(nn.Cell):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args, num_classes, matcher, weight_dict, aux_loss=True):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            args.eos_coef: relative classification weight applied to the no-object category
        """
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = Tensor([args.eos_coef], mstype.float32)

        empty_weight = ops.Ones()(self.num_classes, mstype.float32)
        empty_weight = ops.Concat()([empty_weight, self.eos_coef])

        self.label_weight = weight_dict['loss_ce']
        self.bbox_weight = weight_dict['loss_bbox']
        self.giou_weight = weight_dict['loss_giou']
        self.aux_loss = aux_loss

        self.bbox_loss = BoxLoss()
        self.cls_loss = LogSoftmaxCrossEntropyWithLogits(empty_weight)

        self.reduce_sum = ops.ReduceSum()

    def construct(self, pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valids):
        """
        if aux_loss
            outputs:
                pred_logits: (head, bs, num_queries, num_classes+1)
                pred_boxes: (head, bs, num_queries, 4)
        else:
            outputs:
                pred_logits: (bs, num_queries, num_classes+1)
                pred_boxes: (bs, num_queries, 4)
        targets:
            gt_boxes: (bs, num_queries)
            gt_labels: (bs, num_queries, 4)
            gt_isvalid: (bs, num_queries) [True, True, False, False ......]
        """
        if not self.aux_loss:
            return self.calculate_loss(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valids)
        else:
            losses_1 = self.calculate_loss(pred_logits[0], pred_boxes[0], gt_boxes, gt_labels, gt_valids)
            losses_2 = self.calculate_loss(pred_logits[1], pred_boxes[1], gt_boxes, gt_labels, gt_valids)
            losses_3 = self.calculate_loss(pred_logits[2], pred_boxes[2], gt_boxes, gt_labels, gt_valids)
            losses_4 = self.calculate_loss(pred_logits[3], pred_boxes[3], gt_boxes, gt_labels, gt_valids)
            losses_5 = self.calculate_loss(pred_logits[4], pred_boxes[4], gt_boxes, gt_labels, gt_valids)
            losses_6 = self.calculate_loss(pred_logits[5], pred_boxes[5], gt_boxes, gt_labels, gt_valids)
            return losses_1+losses_2+losses_3+losses_4+losses_5+losses_6

    def calculate_loss(self, pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valids):
        pred_logits = pred_logits.astype(mstype.float32)
        pred_boxes = pred_boxes.astype(mstype.float32)
        target_classes, target_boxes, boxes_valid = self.matcher(pred_logits,
                                                                 pred_boxes,
                                                                 gt_boxes,
                                                                 gt_labels,
                                                                 gt_valids)
        target_classes = ops.stop_gradient(target_classes)
        target_boxes = ops.stop_gradient(target_boxes)
        boxes_valid = ops.stop_gradient(boxes_valid)
        label_losses = self.cls_loss(pred_logits, target_classes)
        loss_bbox, loss_giou = self.bbox_loss(pred_boxes, target_boxes, boxes_valid)
        losses = self.label_weight * label_losses + self.bbox_weight * loss_bbox + self.giou_weight * loss_giou
        return losses
