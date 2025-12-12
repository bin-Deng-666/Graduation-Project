import os
import argparse
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
from typing import List, Dict
from torch import nn
from tqdm import tqdm

# 导入特征提取器
from feature_extractors import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
)

# 骨干网络名称到模型类的映射
BACKBONE_MAP = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}

from utils.attack_tool import (
    load_dataset,
    get_subset
)

class AttackConfig:
    """对抗攻击配置类，集中管理攻击参数"""
    def __init__(self,
                 extractors,
                 datasets,
                 fraction: float = 1.0,
                 output_dir: str = "./output",
                 iters: int = 50,
                 alpha: float = 0.01,
                 epsilon: float = 8.0,
                 device: str = "cuda"):
        """初始化攻击配置
        
        Args:
            extractors: 特征提取器
            datasets: 数据集元组
            fraction: 数据比例
            output_dir: 对抗图像输出目录
            iters: 优化迭代次数
            alpha: 学习率
            epsilon: 最大扰动范围
            device: 运行设备
        """
        self.extractors = extractors
        self.datasets = datasets
        self.fraction = fraction
        self.output_dir = output_dir
        self.iters = iters
        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

def load_feature_extractors(backbones: List[str], device: str = "cuda") -> List[nn.Module]:
    """加载多个特征提取器
    
    Args:
        backbones: 骨干网络名称列表
        device: 运行设备
        
    Returns:
        特征提取器列表
    """
    extractors = []
    for backbone_name in backbones:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(f"未知骨干网络: {backbone_name}")
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(device).requires_grad_(False)
        extractors.append(model)
    return extractors

def get_embeddings(models: List[nn.Module], images: torch.Tensor) -> Dict[int, torch.Tensor]:
    """获取图像在所有模型中的嵌入向量
    
    Args:
        models: 模型列表
        images: 输入图像张量
        
    Returns:
        各模型的嵌入向量字典
    """
    features = {}
    for i, model in enumerate(models):
        features[i] = model(images).squeeze()
    return features

def compute_cosine_distance(features1: Dict[int, torch.Tensor], 
                          features2: Dict[int, torch.Tensor]) -> torch.Tensor:
    """计算两组特征之间的平均余弦距离
    
    Args:
        features1: 第一组特征
        features2: 第二组特征
        
    Returns:
        平均余弦距离
    """
    distance = 0
    for i in features1.keys():
        # 计算余弦相似度
        cos_sim = torch.sum(features1[i] * features2[i], dim=1)
        # 余弦距离 = 1 - 余弦相似度
        cos_distance = 1 - cos_sim
        distance += torch.mean(cos_distance)
    
    return distance / len(features1)

def generate_adversarial_perturbation(image_tensor,
                                      image_collection_paths: List[str],
                                      output_dir: str = "./output",
                                      extractors=None,
                                      iters: int = 50,
                                      alpha: float = 0.01,
                                      epsilon: float = 8.0,
                                      device: str = "cuda"):
    """生成对抗扰动，最大化图像集合与原始图像的嵌入向量距离
    
    Args:
        image_tensor: 原始图像张量
        image_collection_paths: 目标图像集合路径列表
        output_dir: 对抗图像输出目录
        extractors: 特征提取器
        iters: 优化迭代次数
        alpha: 学习率
        epsilon: 最大扰动范围
        device: 运行设备
    """
    # 将图像移至设备
    original_image = image_tensor.to(device)
    
    # 获取原始图像的嵌入向量
    with torch.no_grad():
        original_features = get_embeddings(extractors, original_image)
    
    # 加载图像集合
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 确保图像大小与原始图像一致
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0),  # 归一化到0-255
        transforms.Lambda(lambda x: x.unsqueeze(0))  # 添加batch维度
    ])
    
    collection_images = []
    for img_path in image_collection_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).to(device)
        collection_images.append(img_tensor)
    
    # 初始化可训练扰动参数
    perturbation = torch.zeros_like(original_image, device=device, requires_grad=True)
    
    # 优化过程
    for i in tqdm(range(iters)):
        total_distance = 0
        
        # 使用图像集合中的所有图片加上同一个扰动
        for collection_img in collection_images:
            # 将扰动应用到集合图像上
            perturbed_image = collection_img + perturbation
            # 获取扰动后图像的嵌入向量
            perturbed_features = get_embeddings(extractors, perturbed_image)
            # 计算与原始图像嵌入向量的距离
            distance = compute_cosine_distance(perturbed_features, original_features)
            total_distance += distance
        
        # 计算平均距离
        avg_distance = total_distance / len(collection_images)
        
        # 梯度上升优化（最大化距离）
        if perturbation.grad is not None:
            perturbation.grad.zero_()  # 清零梯度
        
        loss = -avg_distance  # 负距离作为损失函数，梯度上升
        loss.backward()  # 反向传播计算梯度
        
        # 更新扰动并裁剪到合法范围
        with torch.no_grad():
            perturbation.data = torch.clamp(
                perturbation + alpha * torch.sign(perturbation.grad),
                min=-epsilon,
                max=epsilon
            )
        
        # 每10次迭代打印结果
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                # 计算当前扰动下的平均距离
                current_total_distance = 0
                for collection_img in collection_images:
                    perturbed_image = collection_img + perturbation
                    perturbed_features = get_embeddings(extractors, perturbed_image)
                    distance = compute_cosine_distance(perturbed_features, original_features)
                    current_total_distance += distance
                current_avg_distance = current_total_distance / len(collection_images)
                print(f"\n迭代 {i+1}/{iters} 平均距离: {current_avg_distance.item():.4f}")
    
    # 保存结果
    adversarial_image = original_image + perturbation
    
    # 将图像转换为PIL格式以便保存
    transform = transforms.Compose([
        transforms.Lambda(lambda x: (x.cpu().squeeze(0) / 255.0).clamp(0, 1)),
        transforms.ToPILImage()
    ])
    
    original_pil = transform(original_image)
    adversarial_pil = transform(adversarial_image)
    
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图像
    original_pil.save(os.path.join(output_dir, "original.png"))
    adversarial_pil.save(os.path.join(output_dir, "adversarial.png"))
    
    # 保存扰动
    torch.save(perturbation, os.path.join(output_dir, "perturbation.pt"))
    
    print(f"\n结果保存到: {output_dir}")


def print_config(config: AttackConfig) -> None:
    """输出攻击配置参数
    
    Args:
        config: 攻击配置对象
    """
    print(f"\n=== 攻击配置信息 ===")
    print(f"- 输出目录: {config.output_dir}")
    print(f"- 数据集比例: {config.fraction}")
    print(f"- 迭代次数: {config.iters}")
    print(f"- 学习率: {config.alpha}")
    print(f"- 最大扰动: {config.epsilon}")
    print(f"- 设备: {config.device}")
    print(f"- 骨干网络: {', '.join(config.extractors.keys()) if hasattr(config.extractors, 'keys') else '未知'}")
    print(f"- 训练集大小: {len(config.datasets[0])}")
    print(f"- 测试集大小: {len(config.datasets[1])}")


def attack(config: AttackConfig) -> None:
    """主攻击函数，使用配置对象管理参数
    
    Args:
        config: 攻击配置对象
    """
    print(f"\n=== 开始执行对抗攻击 ===")
    # 输出配置参数
    print_config(config)
    
    # 加载数据集子集
    _, test_dataset = config.datasets
    test_dataset = get_subset(test_dataset, config.fraction)
    
    # 生成对抗扰动
    for i, item in enumerate(test_dataset):
        print(f"\n正在处理第 {i+1}/{len(test_dataset)} 个样本...")
        
        # 获取图像张量
        image_tensor = item["image"].to(config.device)

        # 获取图像对应的集合
        
        # 生成对抗扰动（图像集合路径留空）
        generate_adversarial_perturbation(
            image_tensor=image_tensor,
            image_collection_paths=[],  # 图像集合路径留空
            output_dir=config.output_dir,
            extractors=config.extractors,
            iters=config.iters,
            alpha=config.alpha,
            epsilon=config.epsilon,
            device=config.device
        )
    
    print("\n=== 攻击完成 ===")

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对抗攻击参数配置')
    
    parser.add_argument('--backbones', '-b', type=str, nargs='+', default=["B16", "B32"], help='使用的骨干网络列表，可选值: B16, B32, L336, Laion (默认: B16 B32)')
    parser.add_argument('--fraction', '-f', type=float, default=1.0, help='数据集使用比例 (默认: 1.0)')
    parser.add_argument('--iters', '-i', type=int, default=50, help='优化迭代次数 (默认: 50)')
    parser.add_argument('--alpha', '-a', type=float, default=0.01, help='学习率 (默认: 0.01)')
    parser.add_argument('--epsilon', '-e', type=float, default=8.0, help='最大扰动范围 (默认: 8.0)')
    parser.add_argument('--output-dir', '-o', type=str, default="./output", help='对抗图像输出目录 (默认: ./output)')
    parser.add_argument('--device', '-d', type=str, default="cuda", help='运行设备 (默认: cuda)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载图像编码器和数据集
    extractors = load_feature_extractors(backbones=args.backbones, device=args.device)
    train_dataset, test_dataset = load_dataset()

    config = AttackConfig(
        extractors=extractors,
        datasets=(train_dataset, test_dataset),
        fraction=args.fraction,
        output_dir=args.output_dir,
        iters=args.iters,
        alpha=args.alpha,
        epsilon=args.epsilon,
        device=args.device
    )

    # 执行攻击
    attack(config)