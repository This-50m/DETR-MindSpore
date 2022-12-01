import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from mindspore import Tensor
from mindspore import nn
from mindspore import dtype as mstype


def softmax(arr, axis=None):
    """softmax"""
    return np.exp(arr) / np.sum(np.exp(arr), axis=axis, keepdims=True)


def box_xyxy_to_cxcywh(x):
    """box xyxy to cxcywh"""
    x0, y0, x1, y1 = np.array_split(x.T, 4)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)[0]


def box_cxcywh_to_xyxy(x):
    """box cxcywh to xyxy"""
    x_c, y_c, w, h = np.array_split(x, 4, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1).squeeze(-2)


def GIOU(boxes1, boxes2):
    """
    boxes1 shape : shape (n, 4)
    boxes2 shape : shape (k, 4)
    gious: shape (n, k)
    """
    IOU = []
    GIOU = []
    num = (boxes1[:, 0]).size
    x1 = boxes1[:, 0]
    y1 = boxes1[:, 1]
    x2 = boxes1[:, 2]
    y2 = boxes1[:, 3]

    xx1 = boxes2[:, 0]
    yy1 = boxes2[:, 1]
    xx2 = boxes2[:, 2]
    yy2 = boxes2[:, 3]

    area1 = (x2 - x1) * (y2 - y1)  # 求取框的面积
    area2 = (xx2 - xx1) * (yy2 - yy1)
    for i in range(num):
        inter_max_x = np.minimum(x2[i], xx2[:])  # 求取重合的坐标及面积
        inter_max_y = np.minimum(y2[i], yy2[:])
        inter_min_x = np.maximum(x1[i], xx1[:])
        inter_min_y = np.maximum(y1[i], yy1[:])
        inter_w = np.maximum(0, inter_max_x - inter_min_x)
        inter_h = np.maximum(0, inter_max_y - inter_min_y)

        inter_areas = inter_w * inter_h

        out_max_x = np.maximum(x2[i], xx2[:])  # 求取包裹两个框的集合C的坐标及面积
        out_max_y = np.maximum(y2[i], yy2[:])
        out_min_x = np.minimum(x1[i], xx1[:])
        out_min_y = np.minimum(y1[i], yy1[:])
        out_w = np.maximum(0, out_max_x - out_min_x)
        out_h = np.maximum(0, out_max_y - out_min_y)

        outer_areas = out_w * out_h
        union = area1[i] + area2[:] - inter_areas  # 两框的总面积   利用广播机制
        ious = inter_areas / union
        gious = ious - (outer_areas - union) / outer_areas  # IOU - ((C\union）/C)
        IOU.append(ious)
        GIOU.append(gious)
    return np.stack(GIOU, axis=0)


class HungarianMatcherNumpy(nn.Cell):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super(HungarianMatcherNumpy, self).__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def construct(self, pred_logits, pred_boxes, tgt_bbox, tgt_labels, tgt_valid):
        """
        :param pred_logits: (bs, num_queries, num_classes)
        :param pred_boxes: (bs, num_queries, 4)
        :param tgt_bbox: (bs, num_queries, 4)
        :param tgt_labels: (bs, num_queries)
        :param tgt_valid: (bs, num_queries)
        :return:
        """
        # cast to numpy
        pred_logits = pred_logits.asnumpy()
        pred_boxes = pred_boxes.asnumpy()
        tgt_bbox = tgt_bbox.asnumpy().astype(np.float32)
        tgt_labels = tgt_labels.asnumpy().astype(np.int32)
        tgt_valid = tgt_valid.asnumpy().astype(np.bool_)

        bs, num_queries, num_classes = pred_logits.shape

        # We reshape to compute the cost matrices in a batch
        # out_prob [batch_size * num_queries, num_classes]
        out_prob = softmax(pred_logits.reshape(-1, pred_logits.shape[-1]), -1)
        # out_bbox [batch_size * num_queries, 4]
        out_bbox = pred_boxes.reshape(-1, pred_boxes.shape[-1])

        # Also concat the target labels and boxes
        tgt_labels_valid = tgt_labels[tgt_valid]
        tgt_bbox_valid = tgt_bbox[tgt_valid]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_labels_valid]

        # Compute the L1 cost between boxes
        cost_bbox = cdist(out_bbox, tgt_bbox_valid, metric='minkowski', p=1)

        # Compute the giou cost between boxes
        cost_giou = -GIOU(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox_valid))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.reshape(bs, num_queries, -1)

        sizes = np.cumsum(tgt_valid.sum(1))
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(np.split(C, sizes, -1)[:-1])]
        src_idx = np.concatenate([src for (src, _) in indices])
        col_idx = np.concatenate([col for (_, col) in indices])
        batch_idx = np.concatenate([np.full_like(src, i) for i, (src, _) in enumerate(indices)])

        target_classes = np.ones((bs, num_queries)) * (num_classes - 1)  # 91
        target_classes[batch_idx, src_idx] = tgt_labels[batch_idx, col_idx]

        target_boxes = np.zeros((bs, num_queries, 4))
        target_boxes[batch_idx, src_idx] = tgt_bbox[batch_idx, col_idx]

        boxes_valid = np.zeros((bs, num_queries))
        boxes_valid[batch_idx, src_idx] = 1

        target_classes = Tensor(target_classes, dtype=mstype.int32)
        target_boxes = Tensor(target_boxes, dtype=mstype.float32)
        boxes_valid = Tensor(boxes_valid, dtype=mstype.float32)
        return target_classes, target_boxes, boxes_valid


def build_matcher(args):
    return HungarianMatcherNumpy(cost_class=args.set_cost_class,
                                 cost_bbox=args.set_cost_bbox,
                                 cost_giou=args.set_cost_giou)
