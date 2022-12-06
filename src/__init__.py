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


def prepare_args():
    parser = argparse.ArgumentParser('Set transformer detector')

    # training parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--pretrained', default='', type=str, help='resnet_backbone_ckpt')
    parser.add_argument('--seed', default=42, type=int)

    # context
    parser.add_argument('--device_id', default=0, type=int, help='which device')
    parser.add_argument('--device_target', default="GPU", type=str)
    parser.add_argument('--context_mode', default="GRAPH", type=str)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--mindrecord_dir', default='data')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--train_data_type', default='train2017')
    parser.add_argument('--val_data_type', default='val2017')
    parser.add_argument('--num_classes', default=91, type=int, help='90(object) + 1(background)')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--num_parallel_workers', default=8, type=int,
                        help='Number of threads used to process the dataset in parallel')
    parser.add_argument('--python_multiprocessing', action='store_true',
                        help='Parallelize Python operations with multiple worker processes')

    # image processing
    parser.add_argument('--max_size', default=960, type=int)
    parser.add_argument('--flip_ratio', default=0.5, type=float, help='random flip ratio')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")

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

    # distributed switch
    parser.add_argument("--distributed", default=0, type=int, help="is distributed")

    args = parser.parse_args()
    return args
