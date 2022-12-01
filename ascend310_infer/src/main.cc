/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <sys/time.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/dataset/vision.h"
#include "include/dataset/execute.h"
#include "../inc/utils.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::DataType;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::transforms::TypeCast;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::Pad;
using mindspore::dataset::vision::HWC2CHW;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::Decode;

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_int32(MinSize, 800, "image min size");
DEFINE_int32(MaxSize, 1333, "image max size");
DEFINE_int32(TgtSize, 1344, "(max_size/32+1)*32");

void GetSizeWithRatio(const int imgHeight, const int imgWidth,
                      int minSize, int maxSize,
                      int &tgtHeight, int &tgtWidth) {
    float minOriSize = std::min(imgWidth, imgHeight);
    float maxOriSize = std::max(imgWidth, imgHeight);
    if (maxOriSize / minOriSize * minSize > maxSize){
        minSize = round(maxSize * minOriSize / maxOriSize);
    }
    if ((imgWidth <= imgHeight && imgWidth == minSize) || (imgHeight <= imgWidth && imgHeight == minSize)){
        tgtHeight = imgHeight;
        tgtWidth = imgWidth;
    }else{
        if (imgWidth < imgHeight){
            tgtWidth = minSize;
            tgtHeight = minSize * imgHeight / imgWidth;
        } else {
            tgtHeight = minSize;
            tgtWidth = minSize * imgWidth / imgHeight;
        }
    }
}


int PadImageMask(const MSTensor &input, MSTensor *output1, MSTensor *output2) {
    std::vector<int64_t> shape = input.Shape();

    int tgtHeight, tgtWidth;
    GetSizeWithRatio(shape[0], shape[1], FLAGS_MinSize, FLAGS_MaxSize, tgtHeight, tgtWidth);

    Status ret;
    // resize
    auto resize(new Resize({tgtHeight, tgtWidth}));
    auto imgResize = MSTensor();
    Execute composeResizeWidth({resize});
    ret = composeResizeWidth(input, &imgResize);
    if (ret != kSuccess) {
        std::cout << "ERROR: Resize Width failed." << std::endl;
        return 1;
    }

    // normalization
    auto normalize(new Normalize({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}));
    auto imgNorm = MSTensor();
    Execute composeNormalize({normalize});
    ret = composeNormalize(imgResize, &imgNorm);
    if (ret != kSuccess) {
        std::cout << "ERROR: Normalize failed." << std::endl;
        return 1;
    }

    // padding
    int paddingHeightSize = FLAGS_TgtSize - tgtHeight;
    int paddingWidthSize = FLAGS_TgtSize - tgtWidth;

    // left, top, right, and bottom
    auto pad(new Pad({0, 0, paddingWidthSize, paddingHeightSize}));
    Execute composePad({pad});
    ret = composePad(imgNorm, output1);
    if (ret != kSuccess) {
        std::cout << "ERROR: Pad failed." << std::endl;
        return 1;
    }

    // create mask
    int imgMaskInfo[FLAGS_TgtSize][FLAGS_TgtSize];
    for (int i=0; i<FLAGS_TgtSize; i++){
        for (int j=0; j<FLAGS_TgtSize; j++){
            if (i<tgtHeight && j<tgtWidth) {imgMaskInfo[i][j]=0;}
            else {imgMaskInfo[i][j]=1;}
        }
    }
    MSTensor imgMask("imgMask", DataType::kNumberTypeInt32,
                     {static_cast<int64_t>(FLAGS_TgtSize), static_cast<int64_t>(FLAGS_TgtSize)},
                     &imgMaskInfo,
                     FLAGS_TgtSize*FLAGS_TgtSize*4);

    auto typeCast = TypeCast(DataType::kNumberTypeBool);
    Execute transformCast(typeCast);
    ret = transformCast(imgMask, output2);
    if (ret != kSuccess) {
        std::cout << "ERROR: transformCast Mask failed." << std::endl;
        return 1;
    }

    return 0;
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_mindir_path).empty()) {
        std::cout << "Invalid mindir" << std::endl;
        return 1;
    }

    auto context = std::make_shared<Context>();
    auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310->SetDeviceID(FLAGS_device_id);
    ascend310->SetPrecisionMode("allow_fp32_to_fp16");
    context->MutableDeviceInfo().push_back(ascend310);
    // load graph file
    mindspore::Graph graph;
    Serialization::Load(FLAGS_mindir_path, ModelType::kMindIR, &graph);

    // build network
    Model model;
    Status ret = model.Build(GraphCell(graph), context);
    if (ret != kSuccess) {
        std::cout << "ERROR: Build failed." << std::endl;
        return 1;
    }
    std::vector<MSTensor> model_inputs = model.GetInputs();
    if (model_inputs.empty()) {
        std::cout << "Invalid model, inputs is empty." << std::endl;
        return 1;
    }

    auto all_files = GetAllFiles(FLAGS_dataset_path);
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    std::map<double, double> costTime_map;
    size_t size = all_files.size();
    std::shared_ptr<TensorTransform> decode(new Decode());
    Execute composeDecode({decode});
    std::shared_ptr<TensorTransform> hwc2chw(new HWC2CHW());
    Execute composeTranspose({hwc2chw});

    for (size_t i = 0; i < size; ++i) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTimeMs;
        double endTimeMs;
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << all_files[i] << std::endl;

        // ********** start **********
        auto imgDecode = MSTensor();
        auto image = ReadFileToTensor(all_files[i]);
        ret = composeDecode(image, &imgDecode);
        if (ret != kSuccess) {
            std::cout << "ERROR: Decode failed." << std::endl;
            return 1;
        }

        auto imgPad = MSTensor();
        auto imgMask = MSTensor();
        PadImageMask(imgDecode, &imgPad, &imgMask);

        auto img = MSTensor();
        composeTranspose(imgPad, &img);

        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            img.Data().get(), img.DataSize());
        inputs.emplace_back(model_inputs[1].Name(), model_inputs[1].DataType(), model_inputs[1].Shape(),
                            imgMask.Data().get(), imgMask.DataSize());
        // ********** end **********

        gettimeofday(&start, nullptr);
        ret = model.Predict(inputs, &outputs);
        gettimeofday(&end, nullptr);
        if (ret != kSuccess) {
            std::cout << "Predict " << all_files[i] << " failed." << std::endl;
            return 1;
        }
        startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
        WriteResult(all_files[i], outputs);
    }
    double average = 0.0;
    int inferCount = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        inferCount++;
    }
    average = average / inferCount;
    std::stringstream timeCost;
    timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << std::endl;
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;

    std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
    fileStream << timeCost.str();
    fileStream.close();
    costTime_map.clear();
    return 0;
}
