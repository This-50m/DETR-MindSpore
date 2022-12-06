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

if [ $# != 4 ]
then
    echo "Usage: bash scripts/run_distribute_train_ascend.sh [DATASET_PATH] [BACKBONE_PRETRAIN] [CONTEXT_MODE]"
exit 1
fi
DATASET_PATH=$2
BACKBONE_PRETRAIN=$3
CONTEXT_MODE=$4

bootup_cache_server()
{
  echo "Booting up cache server..."
  result=$(cache_admin --start 2>&1)
  rc=$?
  echo "${result}"
  if [ "${rc}" -ne 0 ] && [[ ! ${result} =~ "Cache server is already up and running" ]]; then
    echo "cache_admin command failure!" "${result}"
    exit 1
  fi
}

generate_cache_session()
{
  result=$(cache_admin -g | awk 'END {print $NF}')
  rc=$?
  echo "${result}"
  if [ "${rc}" -ne 0 ]; then
    echo "cache_admin command failure!" "${result}"
    exit 1
  fi
}

shutdown_cache_server()
{
  echo "Shutting down cache server..."
  result=$(cache_admin --stop 2>&1)
  rc=$?
  echo "${result}"
  if [ "${rc}" -ne 0 ] && [[ ! ${result} =~ "Server on port 50052 is not reachable or has been shutdown already" ]]; then
    echo "cache_admin command failure!" "${result}"
    exit 1
  fi
}
export GLOG_v=3

# distributed training json about device ip address
#export RANK_TABLE_FILE=$1
#export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE
# ensure log dir exists
DIR=./outputs
if [[ ! -d "$DIR" ]]; then
    mkdir $DIR
fi

# rank_size: number of device when training
export RANK_SIZE=8
#export DEPLOY_MODE=0

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
	python main.py --coco_path=${DATASET_PATH} \
	                --output_dir=outputs/ --mindrecord_dir=data/ \
	                --clip_max_norm=0.1 \
	                --dropout=0.1 \
	                --batch_size=2 \
	                --pretrained=${BACKBONE_PRETRAIN} \
	                --epochs=300 \
	                --distributed=1 \
	                --context_mode=${CONTEXT_MODE} \
	                --device_target="GPU" &> log &



