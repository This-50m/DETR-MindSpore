#!/bin/bash
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

# set Ascend910 env
source scripts/env_npu.sh;
export GLOG_v=3

if [ $# != 4 ]
then
    echo "Usage: bash scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH] [BACKBONE_PRETRAIN] [CONTEXT_MODE]"
exit 1
fi

# bash scripts/run_distribute_train_ascend.sh hccl_8p_01234567_127.0.0.1.json /opt/npu/data/coco2017 ms_resnet_50.ckpt GRAPH"

# distributed training json about device ip address
export RANK_TABLE_FILE=$1
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE
DATASET_PATH=$2
BACKBONE_PRETRAIN=$3
CONTEXT_MODE=$4

DIR=./outputs
if [[ ! -d "$DIR" ]]; then
    mkdir $DIR
fi

# rank_size: number of device when training
export RANK_SIZE=8
#export DEPLOY_MODE=0

KERNEL_NUM=$(($(nproc)/${RANK_SIZE}))
for((i=0;i<$((RANK_SIZE));i++));
  do
    export RANK_ID=${i}
    echo "start training for device $i rank_id $RANK_ID"
    PID_START=$((KERNEL_NUM*i))
    PID_END=$((PID_START+KERNEL_NUM-1))
    taskset -c ${PID_START}-${PID_END} \
      python main.py --coco_path=${DATASET_PATH} \
               --output_dir=outputs/ \
               --mindrecord_dir=data/ \
               --clip_max_norm=0.1 \
               --no_aux_loss \
               --dropout=0.1 \
               --pretrained=${BACKBONE_PRETRAIN} \
               --epochs=300 \
               --distributed=1 \
               --context_mode=${CONTEXT_MODE} \
               --device_target="Ascend" \
               --device_id=${i} >> outputs/train${i}.log 2>&1 &

  done


