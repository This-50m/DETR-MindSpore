import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.scipy.optimize.linear_sum_assignment import _linear_sum_assignment as lsap
from src.DETR.util import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcherAscend(nn.Cell):
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
        super(HungarianMatcherAscend, self).__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        self.softmax = nn.Softmax(axis=-1)
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.cat = ops.Concat()
        self.transpose = ops.Transpose()
        self.masked_select = ops.MaskedSelect()
        self.scatter_nd_update = ops.ScatterNdUpdate()
        self.scatter_nd = ops.ScatterNd()
        self.stack = ops.Stack()
        self.ones = ops.Ones()
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()
        self.max = ops.ReduceMax()
        self.lsap_maximize = Tensor(False)

        self.cur_target_classes = ms.Parameter(ops.Zeros()((100,), ms.float32))
        self.cur_target_boxes = ms.Parameter(ops.Zeros()((100, 4), ms.float32))
        self.cur_boxes_valid = ms.Parameter(ops.Zeros()((100,), ms.float32))

    def construct(self, pred_logits, pred_boxes, tgt_bbox, tgt_labels, tgt_valid):
        """
        :param pred_logits: (bs, num_queries, num_classes)
        :param pred_boxes: (bs, num_queries, 4)
        :param tgt_bbox: (bs, num_queries, 4)
        :param tgt_labels: (bs, num_queries)
        :param tgt_valid: (bs, num_queries)
        :return:
        """
        bs, num_queries, num_classes = pred_logits.shape
        target_classes = []
        target_boxes = []
        boxes_valid = []

        tgt_valid = tgt_valid.astype(ms.float32)
        # TODO
        for i in range(bs):
            cur_pred_logits = pred_logits[i]
            cur_pred_boxes = pred_boxes[i]
            cur_tgt_bbox = tgt_bbox[i]
            cur_tgt_labels = tgt_labels[i]
            cur_tgt_valid = tgt_valid[i]

            out_prob = self.softmax(cur_pred_logits)

            # cost_class = -out_prob[:, cur_tgt_labels]
            cost_class = -ops.gather(ops.pad(out_prob, ((0, 0), (0, num_queries - num_classes))),
                                     cur_tgt_labels,
                                     axis=1)

            cost_bbox = ops.cdist(cur_pred_boxes, cur_tgt_bbox, 1.0)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(cur_pred_boxes), box_cxcywh_to_xyxy(cur_tgt_bbox))

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            src, col = lsap(C, self.lsap_maximize, cur_tgt_valid.sum().astype(ms.int64))
            src, col = self.reshape(src, (num_queries,)), self.reshape(col, (num_queries,))

            cur_tgt_invalid = 1 - cur_tgt_valid
            src_val = ops.ReduceMax()(src)
            col_val = ops.ReduceMax()(src * 100 + col) - src_val * 100
            src = (src + cur_tgt_invalid * (src_val + 1)).astype(ms.int32)
            col = (col + cur_tgt_invalid * (col_val + 1)).astype(ms.int32)

            # cur_target_classes = self.ones_like(cur_tgt_valid) * (num_classes - 1)
            # cur_target_classes[src] = cur_tgt_labels[col]
            self.cur_target_classes = self.cur_target_classes * 0. + (num_classes - 1)
            gather_labels = ops.gather(cur_tgt_labels, col, axis=0)
            cur_target_classes = ops.scatter_update(self.cur_target_classes, src, gather_labels)

            # cur_target_boxes = self.zeros_like(cur_tgt_bbox)
            # cur_target_boxes[src] = cur_tgt_bbox[col]
            self.cur_target_boxes = self.cur_target_boxes * 0.
            gather_bboxes = ops.gather(cur_tgt_bbox, col, axis=0)
            cur_target_boxes = ops.scatter_update(self.cur_target_boxes, src, gather_bboxes)

            # cur_boxes_valid = self.zeros_like(cur_tgt_valid)
            # cur_boxes_valid[src] = 1
            self.cur_boxes_valid = self.cur_boxes_valid * 0.
            cur_boxes_valid = ops.scatter_update(self.cur_boxes_valid, src, self.ones_like(cur_tgt_valid))

            target_classes.append(cur_target_classes)
            target_boxes.append(cur_target_boxes)
            boxes_valid.append(cur_boxes_valid)

        target_classes = self.stack(target_classes)
        target_boxes = self.stack(target_boxes)
        boxes_valid = self.stack(boxes_valid)
        return target_classes.astype(ms.int32), target_boxes, boxes_valid


def build_matcher(args):
    return HungarianMatcherAscend(cost_class=args.set_cost_class,
                                  cost_bbox=args.set_cost_bbox,
                                  cost_giou=args.set_cost_giou)
