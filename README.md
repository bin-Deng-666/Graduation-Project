# 多模态模型对抗攻击研究

本毕业设计项目实现了两种针对多模态模型的对抗攻击方法，用于研究和评估模型在对抗样本下的鲁棒性。

## 核心功能

### 1. 图像嵌入向量距离最大化攻击 (`maximize_embedding.py`)
通过生成对抗扰动，最大化目标图像集合与原始图像在多个特征提取器（如CLIP系列模型）中的嵌入向量距离，从而降低模型对图像内容的识别能力。

### 2. 文本嵌入对抗攻击 (`train.py`)
实现了多种文本嵌入攻击方法，包括：
- `embed_noise`: 向文本嵌入添加高斯噪声
- `token_noise`: 对输入token添加整数噪声
- `grad_embed_noise`: 使用梯度优化生成对抗嵌入噪声
- `embed_adv`: 生成专门的对抗嵌入序列

## 文件结构

```
GraduationProject/
├── feature_extractors/   # 特征提取器实现（CLIP系列模型）
├── models/              # 多模态评估模型实现
├── utils/               # 工具函数
├── test/                # 测试文件
├── maximize_embedding.py # 图像嵌入距离最大化攻击
├── train.py             # 文本嵌入对抗攻击
├── attack.sbatch        # Slurm攻击脚本
├── evaluate.sbatch      # Slurm评估脚本
├── run_attack.sbatch    # 运行攻击脚本
├── run_evaluate.sh      # 运行评估脚本
└── .gitignore           # Git忽略文件配置
```