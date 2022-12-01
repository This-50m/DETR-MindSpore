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

import torch
from mindspore import Parameter
from collections import OrderedDict
from mindspore import save_checkpoint, load_param_into_net
from src.DETR.resnet import resnet50


def do_backbone(weight_path, save_path):
    pth_dict = torch.load(weight_path, map_location='cpu')
    ckpt_dict = OrderedDict()
    for k, v in pth_dict.items():
        if 'fc' in k:
            continue
        name = k
        if 'bn' in name or 'downsample' in name:
            name = name.replace('running_mean', 'moving_mean')
            name = name.replace('running_var', 'moving_variance')
            name = name.replace('bias', 'beta')
            if 'downsample.0' not in name:
                name = name.replace('weight', 'gamma')
        ckpt_dict[name] = Parameter(v.detach().float().numpy())
    model = resnet50()
    load_param_into_net(model, ckpt_dict, strict_load=True)
    save_checkpoint(model, save_path)
    print('convert over')


def do_sota():
    weight_path = '../../detr_sota.pth'
    pth_dict = torch.load(weight_path, map_location='cpu')['model']
    ckpt_dict = OrderedDict()
    for key, value in pth_dict.items():
        name = key

        if 'self_attn' in name or 'multihead_attn' in name:
            if 'out_proj' in name:
                name = name.replace('out_proj', 'out')
                ckpt_dict[name] = Parameter(value.detach().numpy())
            else:
                pos = name.find('in_proj')
                prefix = name[:pos]
                if 'weight' in name:
                    q_name = prefix + "q_dense.weight"
                    k_name = prefix + "k_dense.weight"
                    v_name = prefix + "v_dense.weight"
                else:
                    q_name = prefix + "q_dense.bias"
                    k_name = prefix + "k_dense.bias"
                    v_name = prefix + "v_dense.bias"
                q, k, v = value.split(value.size(0)//3, 0)
                ckpt_dict[q_name] = Parameter(q.detach().numpy())
                ckpt_dict[k_name] = Parameter(k.detach().numpy())
                ckpt_dict[v_name] = Parameter(v.detach().numpy())
        elif 'norm' in name:
            name = name.replace('weight', 'gamma')
            name = name.replace('bias', 'beta')
            ckpt_dict[name] = Parameter(value.detach().numpy())
        elif 'bbox_embed' in name:
            if '1' in name:
                name = name.replace('1', '2')
            elif '2' in name:
                name = name.replace('2', '4')
            else:
                name = name
            ckpt_dict[name] = Parameter(value.detach().numpy())
        elif 'query_embed' in name:
            name = name.replace('weight', 'embedding_table')
            ckpt_dict[name] = Parameter(value.detach().numpy())
        elif 'backbone' in name:
            name = name.replace('backbone.0', 'backbone.backbone')
            if 'bn' in name or 'downsample' in name:
                name = name.replace('running_mean', 'moving_mean')
                name = name.replace('running_var', 'moving_variance')
                name = name.replace('bias', 'beta')
                if 'downsample.0' not in name:
                    name = name.replace('weight', 'gamma')
            ckpt_dict[name] = Parameter(value.detach().numpy())
        else:
            ckpt_dict[name] = Parameter(value.detach().numpy())

    # args = prepare_args()
    # model, _, _ = build_model(args)
    # param_not_load = load_param_into_net(model, ckpt_dict, strict_load=True)
    # print(param_not_load)
    # save_checkpoint(model, '../../ms_detr_sota.ckpt')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--src', type=str)
    parser.add_argument('--tgt', type=str)
    args = parser.parse_args()
    do_backbone(args.src, args.tgt)
