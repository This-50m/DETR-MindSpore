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

import argparse
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.DETR.backbone import build_backbone
from src.DETR.detr import build_transformer, DETR


def prepare_args():
    parser = argparse.ArgumentParser('Set transformer detector')

    # export main parameters
    parser.add_argument('--resume', default='ms_detr_sota.ckpt', type=str, help='resume from checkpoint')
    parser.add_argument('--pretrained', default='', type=str, help='resnet_backbone_ckpt')
    parser.add_argument('--device_id', default=6, type=int, help='which device')
    parser.add_argument('--device_target', default="GPU", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--file_name', default='detr', type=str)
    parser.add_argument('--file_format', default='MINDIR', choices=["AIR", "MINDIR"], type=str)

    # dataset parameters
    parser.add_argument('--min_size', default=800, type=int)
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--num_classes', default=91, type=int, help='90(object) + 1(background)')

    # * Backbone
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--dice_loss_coef', default=1., type=float)
    parser.add_argument('--bbox_loss_coef', default=5., type=float)
    parser.add_argument('--giou_loss_coef', default=2., type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    args = parser.parse_args()
    return args


def build_net(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )
    return model


def main(args):
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target in ["Ascend", "GPU"]:
        context.set_context(device_id=args.device_id)

    net = build_net(args)
    net.set_train(False)
    load_param_into_net(net, load_checkpoint(args.resume), strict_load=True)

    if args.device_target == "Ascend":
        net.to_float(ms.float16)
        print('cast to float16')

    bs = args.batch_size
    input_arr = Tensor(np.random.rand(bs, 3, args.max_size, args.max_size), ms.float32)
    mask_arr = Tensor(np.zeros([bs, args.max_size, args.max_size]), ms.float32)

    # file_format choose in ["AIR", "MINDIR"]
    export(net, input_arr, mask_arr, file_name=args.file_name, file_format=args.file_format)


if __name__ == '__main__':
    """
    recommend cmd
    >>> python export.py --resume=ms_detr_sota.ckpt \
                 --no_aux_loss \
                 --device_id=3 \
                 --device_target="Ascend" \
                 --batch_size=1 \
                 --file_name='detr_bs1' \
                 --file_format='MINDIR' \
                 --max_size=1280 
    """
    args = prepare_args()
    main(args)
