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
if [ $# != 5 ]
then
    echo "Usage: bash scripts/run_eval_ascend.sh [DATASET_PATH] [RESUME] [DEVICE_TARGET] [DEVICE_ID] [MAX_SIZE]"
exit 1
fi

DATASET_PATH=$1
RESUME=$2
DEVICE_TARGET=$3
DEVICE_ID=$4
MAX_SIZE=$4

python eval.py --coco_path=${DATASET_PATH} \
               --output_dir=outputs/ \
               --mindrecord_dir=data/ \
               --no_aux_loss \
               --device_id=${DEVICE_ID} \
               --device_target=${DEVICE_TARGET} \
               --resume=${RESUME} \
               --max_size=${MAX_SIZE}
