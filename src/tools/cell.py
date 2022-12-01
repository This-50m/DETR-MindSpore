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
from mindspore.context import ParallelMode
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter
import mindspore.ops.functional as F
from mindspore.ops import composite as C
from mindspore import dtype as mstype
from mindspore import context
from mindspore import ms_function
from mindspore.communication.management import get_group_size
from mindspore.parallel._auto_parallel_context import auto_parallel_context


class WithLossCell(nn.Cell):
    def __init__(self, net, criterion):
        super(WithLossCell, self).__init__()
        self.net = net
        self.criterion = criterion

    def construct(self, x, mask, gt_boxes, gt_labels, gt_valids):
        pred_logits, pred_boxes = self.net(x, mask)
        losses = self.criterion(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valids)
        return losses

    @property
    def backbone_network(self):
        return self.net


GRADIENT_CLIP_TYPE = 1
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.Reciprocal()(scale)


class WithGradCellAscend(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, clip_value=0.1):
        # hacker
        scaling_sens = Tensor([0.1], dtype=mstype.float32)
        # manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        super(WithGradCellAscend, self).__init__(network, optimizer, scale_sense=scaling_sens)
        self.max_grad_norm = clip_value
        # hacker
        self.enable_tuple_broaden = True

    def construct(self, *inputs):
        # compute loss
        loss = self.network(*inputs)

        # loss scale
        status, scaling_sens = self.start_overflow_check(loss, self.scale_sense)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, self.weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)

        # apply grad reducer on grads
        if self.reducer_flag:
            grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)

        # if there is no overflow, do optimize
        if not overflow:
            # clip grads
            grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
            loss = F.depend(loss, self.optimizer(grads))
        else:
            print('current gradients is overflowing, skip this step')
        return loss


# ------------------------------------------------------------

class WithGradCell(nn.Cell):
    """train one step cell with sense"""

    def __init__(self, network, optimizer, clip_value=0.1):
        super().__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.scale_sense = Parameter(Tensor(1., dtype=mstype.float32), name="scale_sense")
        self.reducer_flag = False
        self.grad_reducer = None
        self.max_grad_norm = clip_value
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        # this is a hack
        self.enable_tuple_broaden = True

    @ms_function
    def clip_backward(self, loss, grads):
        grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss

    def construct(self, *inputs):
        """construct"""
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs, self.scale_sense)
        return self.clip_backward(loss, grads)
