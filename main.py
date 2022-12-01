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
import time
from collections import deque
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common import set_seed

from src import prepare_args
from src.DETR import build_model
from src.data.dataset import create_mindrecord, create_detr_dataset
from src.tools.cell import WithLossCell, WithGradCell
from src.tools.average_meter import AverageMeter


def main():
    args = prepare_args()

    if args.device_target == 'Ascend':
        # context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    elif args.device_target == 'GPU':
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    else:
        raise ValueError
    # init seed
    set_seed(args.seed)

    # distributed init
    device_num = int(os.getenv('RANK_SIZE', '1'))
    if args.distributed:
        context.set_context(device_id=args.device_id)
        rank = int(os.getenv('RANK_ID', 0))
        print(f'get device_id: {args.device_id}, rank_id: {rank}')
        if args.device_target == "Ascend":
            init(backend_name='hccl')
        else:
            init(backend_name="nccl")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        print(f'distributed init: {args.device_id}/{device_num}')
    else:
        rank = 0
        device_num = 1
        context.set_context(device_id=args.device_id)

    # dataset
    mindrecord_file = create_mindrecord(args, rank, "DETR.mindrecord", True)
    dataset = create_detr_dataset(args, mindrecord_file, batch_size=args.batch_size,
                                  device_num=device_num, rank_id=rank,
                                  num_parallel_workers=args.num_parallel_workers,
                                  python_multiprocessing=args.python_multiprocessing)
    # dataset = build_dataset()
    dataset_size = dataset.get_dataset_size()
    print("Create COCO dataset done!")
    print(f"COCO dataset num: {dataset_size}")

    # model
    net, criterion, postprocessors = build_model(args)
    # load pretrained weights
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        if 'net' in list(ckpt.keys())[0]:
            ckpt = {k[4:]: v for k, v in ckpt.items()}
        load_param_into_net(net, ckpt, strict_load=True)
        print('load pretrained weights checkpoint')

    data_dtype = ms.float32
    if args.device_target == 'Ascend':
        net.to_float(ms.float16)
        for _, cell in net.cells_and_names():
            if isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm)):
                cell.to_float(ms.float32)
        data_dtype = ms.float16
    net.set_train()

    # lr and optimizer
    lr = nn.piecewise_constant_lr(
        [dataset_size * args.lr_drop, dataset_size * args.epochs],
        [args.lr, args.lr * 0.1]
    )
    lr_backbone = nn.piecewise_constant_lr(
        [dataset_size * args.lr_drop, dataset_size * args.epochs],
        [args.lr_backbone, args.lr_backbone * 0.1]
    )
    backbone_params = list(filter(lambda x: 'backbone' in x.name, net.trainable_params()))
    no_backbone_params = list(filter(lambda x: 'backbone' not in x.name, net.trainable_params()))
    param_dicts = [
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': args.weight_decay},
        {'params': no_backbone_params, 'lr': lr, 'weight_decay': args.weight_decay}
    ]
    optimizer = nn.AdamWeightDecay(param_dicts)

    # init mindspore model
    net_with_loss = WithLossCell(net, criterion)
    net_with_grad = WithGradCell(net_with_loss, optimizer, clip_value=args.clip_max_norm)
    print("Create DETR network done!")

    # callbacks
    loss_meter = AverageMeter()
    ckpt_deque = deque()
    data_loader = dataset.create_dict_iterator()
    for e in range(args.start_epoch, args.epochs):
        for i, data in enumerate(data_loader):
            start_time = time.time()
            img_data = data['image'].astype(data_dtype)
            mask = data['mask'].astype(data_dtype)
            boxes = data['boxes']
            labels = data['labels']
            valid = data['valid']
            loss = net_with_grad(img_data, mask, boxes, labels, valid)

            loss_meter.update(loss.asnumpy())
            end_time = time.time()

            if i % (dataset_size//50) == 0:
                fps = args.batch_size / (end_time - start_time)
                print('epoch[{}/{}], iter[{}/{}], loss:{:.4f}, fps:{:.2f} imgs/sec, lr:[{}/{}]'.format(
                    e, args.epochs,
                    i, dataset_size,
                    loss_meter.average(),
                    fps,
                    lr_backbone[e * dataset_size + i], lr[e * dataset_size + i]
                ), flush=True)
        loss_meter.reset()
        if rank == 0:
            ckpt_path = os.path.join('./outputs', f'detr_epoch_{e}.ckpt')
            ms.save_checkpoint(net, ckpt_path)
            if len(ckpt_deque) > 3:
                pre_ckpt_path = ckpt_deque.popleft()
                os.remove(pre_ckpt_path)
            ckpt_deque.append(ckpt_path)


if __name__ == '__main__':
    main()
