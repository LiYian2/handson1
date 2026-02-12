# ResNet on FashionMNIST 完整项目Pipeline

---

## 📋 项目总览

**目标**：复现ResNet并在FashionMNIST-Resplit上进行系统性实验，产出技术报告. 可以参考HA1.md中的要求.

**时间规划**：10-11天

**交付物**：
1. 完整可运行代码库
2. 实验日志和结果数据
3. 技术报告PDF（含图表）

**重要提示**：
本电脑为Macbook Air，M3芯片，内存16GB。请不要尝试在本机器上训练任何模型，只需要完成代码编写和调试即可。所有训练任务均在学校的GPU服务器上完成。
---

## 🏗️ 项目结构设计

```
resnet-fashionmnist/
│
├── README.md                           # 项目说明文档
├── requirements.txt                    # 依赖包列表
│
├── configs/                           # 配置文件
│   ├── base_config.yaml              # 基础配置
│   ├── models/                       # 模型配置
│   │   ├── resnet9.yaml
│   │   ├── resnet18.yaml
│   │   ├── resnet50.yaml
│   │   └── vgg11.yaml
│   └── experiments/                  # 实验配置
│       ├── exp1_baseline.yaml
│       ├── exp2_skip_connection.yaml
│       ├── exp3_batchnorm.yaml
│       ├── exp4_optimizer.yaml
│       ├── exp5_lr.yaml
│       └── exp6_activation.yaml
│
├── data/                              # 数据目录
│   ├── FashionMNIST-Resplit/        # 数据集（已经创建好了）
│   │   ├── data.parquet
│   │   ├── test.csv
│   │   └── train.csv
│   └── transforms.py                 # 数据增强定义
│   └── vis.py                       # 可视化数据(已实现部分功能，你可以用于参考如何读取数据)
│   
├── models/                            # 模型定义
│   ├── __init__.py
│   ├── resnet.py                     # ResNet系列（9/18/50）
│   ├── plainnet.py                   # 无skip connection版本
│   ├── vgg.py                        # VGG baseline
│   └── blocks.py                     # 通用模块（ResidualBlock等）
│
├── utils/                             # 工具函数
│   ├── __init__.py
│   ├── trainer.py                    # 训练器类
│   ├── evaluator.py                  # 评估器类
│   ├── metrics.py                    # 指标计算（准确率、混淆矩阵等）
│   ├── logger.py                     # 日志记录
│   ├── visualization.py              # 可视化工具
│   └── checkpoint.py                 # 模型保存/加载
│
├── experiments/                       # 实验脚本
│   ├── run_experiment.py             # 通用实验运行脚本
│   ├── stage1_baseline_selection.py  # 阶段1：模型选择
│   ├── stage2_ablation_studies.py    # 阶段2：消融实验
│   └── stage3_additional.py          # 阶段3：额外实验
│
├── analysis/                          # 分析脚本
│   ├── compare_models.py             # 模型对比分析
│   ├── plot_training_curves.py       # 绘制训练曲线
│   ├── dead_neurons_analysis.py      # 死神经元分析
│   └── generate_report_figures.py    # 生成报告图表
│
├── outputs/                           # 输出目录
│   ├── checkpoints/                  # 模型检查点
│   ├── logs/                         # 训练日志
│   │   ├── tensorboard/
│   │   └── wandb/
│   ├── results/                      # 实验结果
│   │   ├── exp1_baseline/
│   │   ├── exp2_skip_connection/
│   │   └── ...
│   └── figures/                      # 图表输出
│       ├── training_curves/
│       ├── confusion_matrices/
│       └── comparison_plots/
│
├── report/                            # 报告相关
│   ├── template.tex                  # LaTeX模板
│   ├── figures/                      # 报告图片
│   ├── tables/                       # 报告表格
│   └── main.tex                      # 主报告文件
│
└── tests/                             # 单元测试
    ├── test_models.py
    ├── test_data.py
    └── test_training.py
```

---

## 🔄 详细Pipeline流程

### **Phase 0: 环境准备（Day 0）**

#### 0.1 创建项目环境
在conda中创建虚拟环境并安装依赖包。应该使用的包（python=3.12, pytorch, numpy, torchvision, pandas, seaborn, matplotlib, tqdm, tensorboard, transformers, wandb, datasets）我已经装好了，可以跳过安装。请使用以下命令激活环境：
```bash
# 创建虚拟环境
conda activate py_312
```

#### 0.2 初始化项目结构
1. 创建目录结构
2. 验证数据加载正常
### **Phase 1: 核心代码框架搭建（Day 1-2）**

#### 1.1 配置系统
```yaml
# configs/base_config.yaml
data:
  data_dir: "./data/FashionMNIST-Resplit"
  num_classes: 10
  input_channels: 1
  image_size: 28

training:
  batch_size: 128
  num_epochs: 50
  num_workers: 4
  pin_memory: true
  
  optimizer:
    type: "SGD"
    lr: 0.1
    momentum: 0.9
    weight_decay: 0.0001
  
  lr_scheduler:
    type: "CosineAnnealingLR"
    T_max: 50
  
logging:
  use_wandb: true
  use_tensorboard: true
  log_interval: 10
  save_interval: 5

device: "cuda"
seed: 42
```

#### 1.2 数据处理模块（可以运行）
- 实现`data/transforms.py`，定义数据增强和预处理。
- 实现数据加载器，确保能正确加载FashionMNIST-Resplit数据集（可参考`data/vis.py`）
- 编写单元测试验证数据加载正确性
- 可视化部分样本，确保数据预处理正确

#### 1.3 模型模块（可以运行）
- 实现`models/blocks.py`，定义ResNet的基本模块（ResidualBlock等）
- 实现`models/resnet.py`，定义ResNet9/18/50模型架构
- 实现`models/plainnet.py`，定义无skip connection的baseline
- 实现`models/vgg.py`，定义VGG11 baseline
- 编写单元测试验证模型前向传播正确性
#### 1.4 工具模块（可以运行）
- 实现`utils/trainer.py`，定义训练器类
- 实现`utils/evaluator.py`，定义评估器类
- 实现`utils/metrics.py`，定义指标计算函数
- 实现`utils/logger.py`，定义日志记录功能
- 实现`utils/visualization.py`，定义可视化工具
- 实现`utils/checkpoint.py`，定义模型保存/加载功能
- 编写单元测试验证各工具模块功能正确性
### **Phase 2: 实验脚本开发（Day 3-4）**
#### 2.1 通用实验运行脚本
- 实现`experiments/run_experiment.py`，定义通用实验运行流程
#### 2.2 阶段1：模型选择实验脚本
- 实现`experiments/stage1_baseline_selection.py`，进行ResNet9/18/50和VGG11的对比实验
#### 2.3 阶段2：消融实验脚本
- 实现`experiments/stage2_ablation_studies.py`，使用ResNet9模型，进行skip connection, BatchNorm, 优化器, 学习率调度, 激活函数等消融实验
#### 2.4 阶段3： 额外实验脚本
- 实现`experiments/stage3_additional.py`，进行额外实验（如数据增强, 正则化等）
### **Phase 3: 实验运行与日志记录（Day 5-8）**
#### 3.1 实验运行（不需要完成）
- 在学校GPU服务器上运行各阶段实验脚本
- 使用WandB和TensorBoard记录训练过程
#### 3.2 日志管理
- 定期备份日志和模型检查点
- 确保每个实验都有清晰的日志记录
### **Phase 4: 结果分析与报告撰写（Day 9-10）**
#### 4.1 结果分析脚本（需要完成）
- 实现`analysis/compare_models.py`，对比不同模型和实验结果
- 实现`analysis/plot_training_curves.py`，绘制训练和验证曲线
- 实现`analysis/dead_neurons_analysis.py`，分析死神经元现象
- 实现`analysis/generate_report_figures.py`，生成报告所需图表
#### 4.2 报告撰写
- 使用LaTeX模板撰写技术报告
- 整理实验结果和图表，撰写各章节内容
- 反复修改和润色，确保报告质量
