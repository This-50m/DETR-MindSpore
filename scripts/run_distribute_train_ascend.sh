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

# distributed training json about device ip address
export RANK_TABLE_FILE=$1
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE
# ensure GPU_8P_log_336 dir exists
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
      python main.py --coco_path=/opt/npu/data/coco2017 \
               --output_dir=outputs/ \
               --mindrecord_dir=data/ \
               --clip_max_norm=0.1 \
               --no_aux_loss \
               --dropout=0.1 \
               --pretrained=ms_resnet_50.ckpt \
               --epochs=300 \
               --distributed=1 \
               --context_mode="GRAPH" \
               --device_target="Ascend" \
               --device_id=${i} >> outputs/train${i}.log 2>&1 &

  done


