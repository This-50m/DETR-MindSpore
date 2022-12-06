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

if [[ $# -lt 3 || $# -gt 6 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANNO_PATH] [DEVICE_ID](optional) [IMAGE_WIDTH](optional) [IMAGE_HEIGHT](optional)
    IMAGE_WIDTH, IMAGE_HEIGHT and DEVICE_ID is optional. IMAGE_WIDTH and IMAGE_HEIGHT must be set at the same time
    or not at the same time. IMAGE_WIDTH default value is 1280, IMAGE_HEIGHT default value is 760, DEVICE_ID can be
    set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
data_path=$(get_real_path $2)
anno_path=$(get_real_path $3)
device_id=0
min_size=800
max_size=1280
tgt_size=1280

if [ $# -eq 4 ]; then
    device_id=$4
fi

if [ $# -eq 6 ]; then
    device_id=$4
    min_size=$5
    max_size=$6
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "anno_path: " $anno_path
echo "device id: "$device_id
echo "min_size: "$min_size
echo "max_size: "$max_size
echo "tgt_size: "$tgt_size

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/bin:${ASCEND_HOME}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
    export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_HOME}/ascend-toolkit/latest/opp
    export ASCEND_AICPU_PATH=${ASCEND_HOME}/ascend-toolkit/latest/
    export PYTHONPATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/python/site-packages:${PYTHONPATH}
    export TOOLCHAIN_HOME=${ASCEND_HOME}/ascend-toolkit/latest/toolkit
else
    export PATH=${ASCEND_HOME}/latest/compiler/bin:${ASCEND_HOME}/latest/compiler/ccec_compiler/bin:${PATH}
    export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/latest/lib64:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
    export ASCEND_AICPU_PATH=${ASCEND_HOME}/latest
    export PYTHONPATH=${ASCEND_HOME}/latest/compiler/python/site-packages:${PYTHONPATH}
    export TOOLCHAIN_HOME=${ASCEND_HOME}/latest/toolkit
fi

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
    cd - || exit
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../ascend310_infer/out/main --mindir_path=$model \
                                --dataset_path=$data_path \
                                --device_id=$device_id \
                                --MinSize=$min_size \
                                --MaxSize=$max_size \
                                --TgtSize=$tgt_size &> infer.log
}

function cal_map()
{
    python ../postprocess.py --anno_path=$anno_path --result_dir=./result_Files &> map.log &
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo "execute inference failed"
    exit 1
fi
cal_map
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi

cat map.log | grep ap