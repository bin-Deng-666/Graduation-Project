import os
import argparse
import torchvision.transforms as transforms
import torch
import torchvision
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

class AttackConfig:
    """对抗攻击配置类，集中管理攻击参数"""
    def __init__(self,
                 original_image_path: str,
                 image_collection_paths: List[str],
                 output_dir: str = "./output",
                 backbones: List[str] = ["B16", "B32"],
                 iters: int = 50,
                 alpha: float = 0.01,
                 epsilon: float = 8.0,
                 device: str = "cuda"):
        """初始化攻击配置
        
        Args:
            original_image_path: 原始图像路径
            image_collection_paths: 目标图像集合路径列表
            output_dir: 对抗图像输出目录
            backbones: 使用的骨干网络列表
            iters: 优化迭代次数
            alpha: 学习率
            epsilon: 最大扰动范围
            device: 运行设备
        """
        self.original_image_path = original_image_path
        self.image_collection_paths = image_collection_paths
        self.output_dir = output_dir
        self.backbones = backbones
        self.iters = iters
        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device
        
        # 验证参数有效性
        self._validate_params()
        
    def _validate_params(self):
        """验证输入参数的有效性"""
        # 检查原始图像路径
        if not os.path.exists(self.original_image_path) or not os.path.isfile(self.original_image_path):
            raise ValueError(f"原始图像路径不存在或不是文件: {self.original_image_path}")
        
        # 检查图像集合路径
        if not self.image_collection_paths:
            raise ValueError("图像集合路径列表不能为空")
        
        for path in self.image_collection_paths:
            if not os.path.exists(path) or not os.path.isfile(path):
                raise ValueError(f"图像集合中的路径不存在或不是文件: {path}")
        
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

def generate_adversarial_perturbation(original_image_path: str, 
                                      image_collection_paths: List[str],
                                      output_dir: str = "./output",
                                      backbones: List[str] = ["B16", "B32"],
                                      iters: int = 50,
                                      alpha: float = 0.01,
                                      epsilon: float = 8.0,
                                      device: str = "cuda"):
    """生成对抗扰动，最大化图像集合与原始图像的嵌入向量距离
    
    Args:
        original_image_path: 原始图像路径
        image_collection_paths: 目标图像集合路径列表
        output_dir: 对抗图像输出目录
        backbones: 使用的骨干网络列表
        iters: 优化迭代次数
        alpha: 学习率
        epsilon: 最大扰动范围
        device: 运行设备
    """
    # 设置图像变换
    transform_fn = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img * 255.0),  # 转换为0-255范围
    ])
    
    # 加载原始图像
    original_img = Image.open(original_image_path)
    original_tensor = transform_fn(original_img).unsqueeze(0)
    
    # 加载图像集合
    image_collection = []
    for path in image_collection_paths:
        img = Image.open(path)
        tensor = transform_fn(img).unsqueeze(0)
        image_collection.append(tensor)
    
    # 加载特征提取器
    models = load_feature_extractors(backbones, device)
    
    # 将图像移至设备
    original_image = original_tensor.to(device)
    image_collection = [img.to(device) for img in image_collection]
    
    # 获取原始图像的嵌入向量
    with torch.no_grad():
        original_features = get_embeddings(models, original_image)
    
    # 初始化可训练扰动参数
    perturbation = torch.zeros_like(original_image, device=device, requires_grad=True)
    
    # 优化过程
    for i in tqdm(range(iters)):
        total_distance = 0
        
        # 遍历图像集合计算总距离
        for idx, img in enumerate(image_collection):
            perturbed_image = img + perturbation
            perturbed_features = get_embeddings(models, perturbed_image)
            distance = compute_cosine_distance(perturbed_features, original_features)
            total_distance += distance
        
        # 计算平均距离
        avg_distance = total_distance / len(image_collection)
        
        # 梯度上升优化（最大化距离）
        if perturbation.grad is not None:
            perturbation.grad.zero_()  # 清零梯度
        
        loss = -avg_distance  # 负距离作为损失函数
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
                distances = []
                for img in image_collection:
                    perturbed_image = img + perturbation
                    perturbed_features = get_embeddings(models, perturbed_image)
                    distance = compute_cosine_distance(perturbed_features, original_features)
                    distances.append(distance.item())
                avg_distance = sum(distances) / len(distances)
                print(f"\n迭代 {i+1}/{iters} 平均距离: {avg_distance:.4f}")
    
    # 生成对抗图像并转换回0-1范围
    adversarial_images = []
    for img in image_collection:
        perturbed_image = img + perturbation
        perturbed_image = torch.clamp(perturbed_image, 0.0, 255.0)
        adversarial_images.append(perturbed_image / 255.0)
    
    # 保存对抗图像
    for i, adversarial_img in enumerate(adversarial_images):
        output_path = os.path.join(output_dir, f"adversarial_{i}.png")
        torchvision.utils.save_image(adversarial_img, output_path)
        print(f"保存对抗图像到: {output_path}")
    
    # 计算并显示距离变化
    with torch.no_grad():
        print("\n原始距离:")
        for i, img_tensor in enumerate(image_collection):
            img_features = get_embeddings(models, img_tensor / 255.0)
            distance = compute_cosine_distance(img_features, original_features)
            print(f"图像 {i}: {distance.item():.4f}")
        
        print("\n对抗后的距离:")
        for i, adv_img in enumerate(adversarial_images):
            adv_features = get_embeddings(models, adv_img * 255.0)
            distance = compute_cosine_distance(adv_features, original_features)
            print(f"图像 {i}: {distance.item():.4f}")


def attack(config: AttackConfig) -> None:
    """主攻击函数，使用配置对象管理参数
    
    Args:
        config: 攻击配置对象
    """
    print(f"\n=== 开始执行对抗攻击 ===")
    print(f"攻击配置:")
    print(f"- 原始图像: {config.original_image_path}")
    print(f"- 图像集合大小: {len(config.image_collection_paths)}张图像")
    print(f"- 输出目录: {config.output_dir}")
    print(f"- 骨干网络: {', '.join(config.backbones)}")
    print(f"- 迭代次数: {config.iters}")
    print(f"- 学习率: {config.alpha}")
    print(f"- 最大扰动: {config.epsilon}")
    print(f"- 设备: {config.device}")
    
    # 生成对抗扰动
    generate_adversarial_perturbation(
        original_image_path=config.original_image_path,
        image_collection_paths=config.image_collection_paths,
        output_dir=config.output_dir,
        backbones=config.backbones,
        iters=config.iters,
        alpha=config.alpha,
        epsilon=config.epsilon,
        device=config.device
    )
    
    print("\n=== 攻击完成 ===")

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对抗攻击参数配置')
    
    # 必需参数
    parser.add_argument('--original-image', '-o', type=str, required=True, 
                        help='原始图像路径')
    parser.add_argument('--image-collection', '-c', type=str, nargs='+', required=True, 
                        help='图像集合路径列表')
    
    # 可选参数
    parser.add_argument('--output-dir', '-od', type=str, default="./output", 
                        help='输出目录 (默认: ./output)')
    parser.add_argument('--backbones', '-b', type=str, nargs='+', default=["B16", "B32"], 
                        help='使用的骨干网络列表，可选值: B16, B32, L336, Laion (默认: B16 B32)')
    parser.add_argument('--iters', '-i', type=int, default=50, 
                        help='优化迭代次数 (默认: 50)')
    parser.add_argument('--alpha', '-a', type=float, default=0.01, 
                        help='学习率 (默认: 0.01)')
    parser.add_argument('--epsilon', '-e', type=float, default=8.0, 
                        help='最大扰动范围 (默认: 8.0)')
    parser.add_argument('--device', '-d', type=str, default="cuda", 
                        help='运行设备 (默认: cuda)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建攻击配置
    config = AttackConfig(
        original_image_path=args.original_image,
        image_collection_paths=args.image_collection,
        output_dir=args.output_dir,
        backbones=args.backbones,
        iters=args.iters,
        alpha=args.alpha,
        epsilon=args.epsilon,
        device=args.device
    )
    
    # 执行攻击
    attack(config)