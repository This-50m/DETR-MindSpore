# 目录

- [Contents](#contents)
  - [DETR description](#detr-description)
  - [Model architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Environment requirements](#environment-requirements)
  - [Quick start](#quick-start)
  - [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
    - [Export MINDIR](#export-mindir)
  - [Model Description](#model-description)
    - [Training Performance on GPU](#training-performance-gpu)
  - [Description of Random Situation](#description-of-random-situation)
  - [ModelZoo Homepage](#modelzoo-homepage)

## [DETR描述](#contents)

DEtection TRansformer，由Facebook提出，将 Transformer 成功整合为检测 pipeline 中心构建块的目标检测框架，没有NMS后处理步骤、没有anchor的，端到端目标检测网络。

> [Paper](https://arxiv.org/abs/2005.12872):  End-to-End Object Detection with Transformers.
> Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko, 2020.

## [模型架构](#contents)

DETR将输入图像首先通过ResNet特征提取模块，提取特征信息，然后结合位置信息进行编码，送入Transformer模块进行Encode&Decode，两个模块的特征维度均为512、中间传播维度均为2048、采用8头注意力模块、堆叠次数均为6层、中间激活函数为relu、dropout概率为0.1。对特征进行编码解码得到高纬特征向量后，进行类别编码和回归框编码，得到类别概率和类别回归框信息。在训练时，针对不同的目标，使用匈牙利算法将其与类别概率和类别回归框信息进行匹配，再去计算相应的loss。

## [数据集](#contents)

使用的数据集： [COCO 2017](https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2F)

- 数据集大小：19 GB
  - 训练集：18 GB，118000张图片
  - 验证集：1GB，5000张图片
  - 标注：241 MB，包含实例，字幕，person_keypoints等
- 数据格式：图片和json文件
  - 标注：数据在dataset.py中处理。

- 目录结构如下所示:

```text
.
├── annotations  # 标注jsons
├── train2017    # 训练数据集
└── val2017      # 推理数据集
```

## [环境要求](#contents)

- 硬件（Ascend处理器）
  - 准备Ascend处理器搭建硬件环境。
- 框架
  - [MindSpore](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/docs/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速开始](#contents)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

### [Ascend处理器环境运行](#contents)

#### 训练

```shell
# standalone train
bash ./scripts/run_standalone_train_gpu.sh

# distribute train
bash ./scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [CFG_PATH] [SAVE_PATH] [BACKBONE_PRETRAIN] [DATASET_PATH]
```

Example:

```shell
# standalone train
# DEVICE_ID - device number for training
# CFG_PATH - path to config
# SAVE_PATH - path to save logs and checkpoints
# BACKBONE_PRETRAIN - path to pretrained backbone
# DATASET_PATH - path to COCO dataset
bash ./scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/resnet50_pretrain.ckpt /path/to/coco

# distribute train (8p)
# DEVICE_NUM - number of devices for training
# other parameters as for standalone train
bash ./scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/resnet50_pretrain.ckpt /path/to/coco
```

#### 验证

```shell
# evaluate
bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [CKPT_PATH] [DATASET_PATH]
```

Example:

```shell
# evaluate
# DEVICE_ID - device number for evaluating
# CFG_PATH - path to config
# SAVE_PATH - path to save logs
# CKPT_PATH - path to ckpt for evaluation
# DATASET_PATH - path to COCO dataset
bash ./scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/ckpt /path/to/coco  
```

## [脚本说明](#contents)

### [脚本及样例代码](#contents)

```text
|- ascend310_infer                         # 310推理代码
|- scripts                                  
|    |- env_npu.sh                         # npu环境变量
|    |- run_distribute_train_ascend.sh     # 8pNPU训练 
|    |- run_distribute_train_gpu.sh        # 8pGPU训练
|    |- run_eval_ascend.sh                 # 单卡验证
|    |- run_infer_310.sh                   # 310推理
|    |- run_standalone_train_ascend.sh     # 单卡NPU、GPU训练
|- src
|    |- data
|    |    |- __init__.py 
|    |    |- coco_eval.py                  # coco验证脚本
|    |    |- dataset.py                    # 数据集加载脚本
|    |    |- transform.py                  # 数据增强脚本
|    |- DETR
|    |    |- __init__.py
|    |    |- backbone.py                   # 骨干网络实现脚本
|    |    |- criterion.py                  # 损失函数实现脚本
|    |    |- detr.py                       # detr模型
|    |    |- init_weights.py               # 模型参数初始化
|    |    |- matcher.py                    # 匈牙利算法实现脚本
|    |    |- matcher_np.py                 # 匈牙利算法np实现脚本 
|    |    |- position_encoding.py          # 位置编码实现脚本
|    |    |- resnet.py                     # resnet实现脚本
|    |    |- transformer.py                # transformer实现脚本
|    |    |- util.py                       # 其他工具
|    |- tools
|    |    |- __init__.py
|    |    |- average_meter.py              # 平均值
|    |    |- cell.py                       # 训练cell脚本
|    |    |- hccl_tools.py                 # 分布式脚本
|    |    |- merge_hccl.py                 # 分布式合并脚本
|    |    |- pth2ckpt.py                   # torch权重迁移脚本
|    |- __init__.py                        # 初始化脚本
|- eval.py                                 # 验证脚本
|- export.py                               # 模型导出脚本
|- infer.py                                # 推理脚本
|- main.py                                 # 训练脚本
|- postprocess.py                          # 310后处理脚本
|- requirements.txt                        # 依赖环境
```

### [脚本参数](#contents)

```text
"lr": 0.0001,                                   # learning rate
"lr_backbone": 0.00001,                         # learning rate for pretrained backbone
"epochs": 300,                                  # number of training epochs
"lr_drop": 200,                                 # epoch`s number for decay lr
"weight_decay": 0.0001,                         # weight decay
"batch_size": 4,                                # batch size
"clip_max_norm": 0.1,                           # max norm of gradients
"max_size": 960                             	# max image size
```

### [训练过程](#contents)

#### [Ascend上训练](#contents)

##### 分布式训练 (8p)

```shell
# DEVICE_NUM - number of devices for training (8)
# other parameters as for standalone train
bash ./scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/resnet50_pretrain.ckpt /path/to/coco
```

Logs will be saved to `/path/to/outputs/train0.log`

Result:

```text
...

...
```

### [评估过程](#contents)

#### Ascend评估

```shell
bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [CKPT_PATH] [DATASET_PATH]
```

Example:

```shell
# DEVICE_ID - device number for evaluating (0)
# CFG_PATH - path to config (./default_config.yaml)
# SAVE_PATH - path to save logs (/path/to/output)
# CKPT_PATH - path to ckpt for evaluation (/path/to/ckpt)
# DATASET_PATH - path to COCO dataset (/path/to/coco)
bash ./scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/ckpt /path/to/coco
```

Logs will be saved to `/path/to/output/log_eval.txt`.

Result:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.620
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.430
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.193
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.607
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.329
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.564
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.802
```

### [导出MINDIR](#contents)

If you want to infer the network on Ascend 310, you should convert the model to MINDIR.

#### GPU

```shell
python export.py --resume=ms_detr_sota.ckpt \
                 --no_aux_loss \
                 --device_id=3 \
                 --device_target="Ascend" \
                 --batch_size=1 \
                 --file_name='detr_bs1' \
                 --file_format='MINDIR'
```

Logs will be saved to parent dir of ckpt, converted model will have the same name as ckpt except extension.

## [模型描述](#contents)

### [性能](#contents)

| 参数           | DETR (8p)                                                    |
| -------------- | ------------------------------------------------------------ |
| 资源           | 8x Nvidia RTX 3090                                           |
| 上传日期       | 15.03.2022                                                   |
| Mindspore 版本 | 1.5.0                                                        |
| 数据集         | COCO2017                                                     |
| 训练参数       | epoch=300, lr=0.0001, lr_backbone=0.00001, weight_decay=0.0001, batch_size=4 |
| 优化器         | AdamWeightDecay                                              |
| 损失函数       | L1, GIOU loss, SoftmaxCrossEntropyWithLogits                 |
| 速度           | fps                                                          |
| mAP0.5:0.95    |                                                              |
| mAP0.5         |                                                              |

## [随机情况说明](#contents)

在main.py中，我们设置了“create_dataset”函数内的种子。同时还使用了train.py中的随机种子。

## [ModelZoo 主页](#contents)

请浏览官网[主页](https://gitee.com/mindspore/models)。
