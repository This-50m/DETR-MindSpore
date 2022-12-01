# DETR-MindSpore



目录说明

```python
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

