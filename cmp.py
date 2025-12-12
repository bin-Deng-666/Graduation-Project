import os
import random
from typing import Tuple, Dict, Any, Optional
from collections import deque

import importlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse

from models import BaseEvalModel
from utils.attack_tool import (
    load_model,
    load_dataset,
    get_subset,
    get_intended_token_ids,
    get_img_id_train_prompt_map,
    get_img_id_environment_map,
    RoundWithSTE
)

class AttackConfig:
    """对抗攻击配置类，用于集中管理所有参数"""
    def __init__(self,
                 method: str,
                 target_text: str,
                 adversarial_length: int,
                 eval_model: BaseEvalModel,
                 datasets: Tuple[Dataset, Dataset],
                 fraction: float,
                 prompt_num: int,
                 iters: int,
                 device: int,
                 epsilon: float = 32/255,
                 alpha: float = 1/255,
                 debug: bool = False):
        self.method = method
        self.target_text = target_text
        self.adversarial_length = adversarial_length
        self.eval_model = eval_model
        self.datasets = datasets
        self.fraction = fraction
        self.prompt_num = prompt_num
        self.iters = iters
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.debug = debug
        self.processor = eval_model.processor


def build_inputs_for_baseline_and_multi_prompt(
    current_inputs: Dict[str, torch.Tensor],
    current_target: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """为baseline和multi_prompt方法构建输入和标签"""
    inputs = {
        'input_ids': current_inputs.input_ids,
        'attention_mask': current_inputs.attention_mask
    }
    labels = get_intended_token_ids(inputs['input_ids'], current_target)
    return inputs, labels


def build_inputs_for_embed_noise(
    question_inputs: Dict[str, torch.Tensor],
    answer_inputs: torch.Tensor,
    question_embeddings: torch.Tensor,
    answer_embeddings: torch.Tensor,
    current_target: torch.Tensor,
    noise_scale: float = 0.3
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """为embed_noise方法构建输入、标签和嵌入"""
    # 添加高斯噪声
    noisy_question_embeddings = question_embeddings + torch.randn_like(question_embeddings) * noise_scale

    # 拼接带噪声的问题嵌入和原始答案嵌入
    combined_embeddings = torch.cat([noisy_question_embeddings, answer_embeddings], dim=1)

    # 构建input_ids和attention_mask
    combined_input_ids = torch.cat([question_inputs.input_ids, answer_inputs], dim=1)
    combined_attention_mask = torch.cat([
        question_inputs.attention_mask,
        torch.ones_like(answer_inputs)
    ], dim=1)
    inputs = {
        'input_ids': combined_input_ids,
        'attention_mask': combined_attention_mask
    }

    # 处理labels
    labels = get_intended_token_ids(inputs['input_ids'], current_target)
    return inputs, labels, combined_embeddings


def build_inputs_for_token_noise(
    question_inputs: Dict[str, torch.Tensor],
    answer_inputs: torch.Tensor,
    current_target: torch.Tensor,
    processor: Any,
    perturb_prob: float = 0.3,
    max_perturb: int = 2
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """为token_noise方法构建输入和标签"""
    # 对input_ids添加整数噪声
    noisy_input_ids = question_inputs.input_ids.clone()
    # 生成扰动掩码和扰动值
    perturb_mask = torch.rand_like(noisy_input_ids.float()) < perturb_prob
    perturb_amount = torch.randint(-max_perturb, max_perturb+1, noisy_input_ids.shape, device=noisy_input_ids.device)
    # 应用扰动
    noisy_input_ids = noisy_input_ids + (perturb_mask.long() * perturb_amount)
    # 确保token ID在有效范围内
    vocab_size = processor.tokenizer.vocab_size
    noisy_input_ids = torch.clamp(noisy_input_ids, min=0, max=vocab_size-1)
    
    # 构建input_ids和attention_mask
    combined_input_ids = torch.cat([noisy_input_ids, answer_inputs], dim=1)
    combined_attention_mask = torch.cat([
        question_inputs.attention_mask,
        torch.ones_like(answer_inputs)
    ], dim=1)
    inputs = {
        'input_ids': combined_input_ids,
        'attention_mask': combined_attention_mask
    }

    # 处理labels
    labels = get_intended_token_ids(inputs['input_ids'], current_target)
    return inputs, labels


def build_inputs_for_grad_embed_noise(
    question_inputs: Dict[str, torch.Tensor],
    answer_inputs: torch.Tensor,
    question_embeddings: torch.Tensor,
    answer_embeddings: torch.Tensor,
    current_target: torch.Tensor,
    text_noise: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """为grad_embed_noise方法构建输入、标签和嵌入"""
    # 使用预先生成的可优化噪声
    noisy_question_embeddings = question_embeddings + text_noise
    
    # 拼接带噪声的问题嵌入和原始答案嵌入
    combined_embeddings = torch.cat([noisy_question_embeddings, answer_embeddings], dim=1)

    # 构建input_ids和attention_mask
    combined_input_ids = torch.cat([question_inputs.input_ids, answer_inputs], dim=1)
    combined_attention_mask = torch.cat([
        question_inputs.attention_mask,
        torch.ones_like(answer_inputs)
    ], dim=1)
    inputs = {
        'input_ids': combined_input_ids,
        'attention_mask': combined_attention_mask
    }

    # 处理labels
    labels = get_intended_token_ids(inputs['input_ids'], current_target)
    return inputs, labels, combined_embeddings


def build_inputs_for_embed_adv(
    question_inputs: Dict[str, torch.Tensor],
    answer_inputs: torch.Tensor,
    question_embeddings: torch.Tensor,
    answer_embeddings: torch.Tensor,
    current_target: torch.Tensor,
    adversarial_embeddings: torch.Tensor,
    adversarial_length: int,
    processor: Any
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """为embed_adv方法构建输入、标签和嵌入"""
    # 拼接embeddings
    combined_embeddings = torch.cat([
        question_embeddings,
        adversarial_embeddings,
        answer_embeddings
    ], dim=1)

    # 构建input_ids和attention_mask
    pad_token_id = processor.tokenizer.pad_token_id
    padded_input_ids = torch.cat([
        question_inputs.input_ids,
        torch.full((1, adversarial_length), pad_token_id, device=question_inputs.input_ids.device),
        answer_inputs
    ], dim=1)
    combined_attention_mask = torch.cat([
        question_inputs.attention_mask,
        torch.ones((1, adversarial_length), device=question_inputs.input_ids.device),
        torch.ones_like(answer_inputs)
    ], dim=1)
    inputs = {
        'input_ids': padded_input_ids,
        'attention_mask': combined_attention_mask
    }

    # 处理labels
    labels = get_intended_token_ids(inputs['input_ids'], current_target)
    return inputs, labels, combined_embeddings


def build_inputs_for_token_adv(
    question_inputs: Dict[str, torch.Tensor],
    answer_inputs: torch.Tensor,
    current_target: torch.Tensor,
    adversarial_token_ids: torch.Tensor,
    adversarial_length: int
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """为token_adv方法构建输入和标签"""
    # 构建input_ids和attention_mask
    rounded_token_ids = RoundWithSTE.apply(adversarial_token_ids)
    combined_token_ids = torch.cat([
        question_inputs.input_ids,
        rounded_token_ids.unsqueeze(0),  # 对抗token
        answer_inputs
    ], dim=1)
    combined_attention_mask = torch.cat([
        question_inputs.attention_mask,
        torch.ones((1, adversarial_length), device=question_inputs.input_ids.device),
        torch.ones_like(answer_inputs)
    ], dim=1)
    inputs = {
        'input_ids': combined_token_ids,
        'attention_mask': combined_attention_mask
    }

    # 处理labels
    labels = get_intended_token_ids(inputs['input_ids'], current_target)
    return inputs, labels


def update_perturbations(
    config: AttackConfig,
    loss: torch.Tensor,
    image_perturbation: torch.Tensor,
    best_loss: torch.Tensor,
    best_attack: Optional[torch.Tensor],
    method_specific_vars: Dict[str, Any],
    ep: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """更新所有扰动参数"""
    # 反向传播
    loss.backward()
    
    # 更新最佳攻击
    if loss < best_loss:
        best_loss = loss
        best_attack = image_perturbation.clone().detach()

    # 更新图像扰动
    grad_img = image_perturbation.grad.detach()
    image_perturbation.data = torch.clamp(
        image_perturbation.data - config.alpha * torch.sign(grad_img),
        min=-config.epsilon,  # 像素值约束
        max=config.epsilon
    )
    # 梯度清零
    image_perturbation.grad.zero_()

    # 根据方法更新特定参数
    if config.method == "embed_adv":
        adversarial_embeddings = method_specific_vars['adversarial_embeddings']
        adversarial_embeddings_init = method_specific_vars['adversarial_embeddings_init']
        
        # 更新文本扰动
        grad_embeddings = adversarial_embeddings.grad.detach()
        update = config.alpha * torch.sign(grad_embeddings) * (1 - ep/config.iters)
        adversarial_embeddings.data = torch.clamp(
            adversarial_embeddings.data + update,
            min=adversarial_embeddings_init - 1,  # 语义保持约束
            max=adversarial_embeddings_init + 1
        )
        adversarial_embeddings.grad.zero_()
        
        method_specific_vars['adversarial_embeddings'] = adversarial_embeddings
        
    elif config.method == "token_adv":
        adversarial_token_ids = method_specific_vars['adversarial_token_ids']
        token_ids_init = method_specific_vars['token_ids_init']
        
        # 更新token_ids
        grad_token_ids = adversarial_token_ids.grad.detach()
        # 更新 token_ids（一步梯度更新 + 约束）
        adversarial_token_ids.data = token_ids_init + torch.clamp(
            adversarial_token_ids.data - token_ids_init + config.alpha * grad_token_ids.sign(),
            min=-2.0,  # 单步最大变化幅度
            max=2.0
        )
        # 确保 token_ids 在有效范围内（0 到 vocab_size-1）
        vocab_size = config.processor.tokenizer.vocab_size
        adversarial_token_ids.data = torch.clamp(
            adversarial_token_ids.data,
            min=0,
            max=vocab_size - 1
        )
        adversarial_token_ids.grad.zero_()
        
        method_specific_vars['adversarial_token_ids'] = adversarial_token_ids
        
    elif config.method == "grad_embed_noise":
        total_text_noise = method_specific_vars['total_text_noise']
        text_idx = method_specific_vars['text_idx']
        text_noise = total_text_noise[text_idx]
        
        # 更新文本噪声
        grad_noise = text_noise.grad.detach()
        text_noise.data = torch.clamp(
            text_noise.data + config.alpha * grad_noise.sign() * (1 - ep/config.iters),
            min=-0.5,
            max=0.5
        )
        text_noise.grad.zero_()
        
        # 将更新后的噪声写回列表
        total_text_noise[text_idx] = text_noise
        method_specific_vars['total_text_noise'] = total_text_noise

    return best_loss, best_attack, method_specific_vars


def save_adversarial_results(
    config: AttackConfig,
    input_x_original: torch.Tensor,
    image_perturbation: torch.Tensor,
    item: Dict[str, Any]
) -> None:
    """保存对抗攻击结果"""
    # 创建保存目录
    output_dir = os.path.join("adversarial_images", f"{config.method}_p{config.prompt_num}")
    os.makedirs(output_dir, exist_ok=True)

    # 生成对抗图像和扰动
    adversarial_image = input_x_original + image_perturbation
    adversarial_image = adversarial_image.squeeze(0)
    perturbation = image_perturbation.squeeze(0)

    # 将张量转换为PIL图像并保存(可视化用)
    adversarial_image_np = adversarial_image.detach().cpu().numpy()
    adversarial_image_np = np.transpose(adversarial_image_np, (1, 2, 0))  # CHW->HWC
    adversarial_image_np = np.clip(adversarial_image_np * 255, 0, 255).astype(np.uint8)
    adv_img = Image.fromarray(adversarial_image_np)

    # 保存原始图像
    original_image = input_x_original.squeeze(0).detach().cpu().numpy()
    original_image = np.transpose(original_image, (1, 2, 0))
    original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)
    orig_img = Image.fromarray(original_image)

    # 保存对抗扰动可视化
    perturbation_np = perturbation.detach().cpu().numpy()
    perturbation_np = np.transpose(perturbation_np, (1, 2, 0))  # CHW->HWC
    # 归一化到0-255范围以便可视化
    perturbation_np = ((perturbation_np - perturbation_np.min()) * 
                     (255/(perturbation_np.max()-perturbation_np.min()))).astype(np.uint8)
    pert_img = Image.fromarray(perturbation_np)

    # 为每个问题创建独立文件夹
    img_id = str(item["image_id"])
    img_dir = os.path.join(output_dir, img_id)
    os.makedirs(img_dir, exist_ok=True)
                
    # 保存所有文件
    orig_img.save(f"{img_dir}/original.png")
    adv_img.save(f"{img_dir}/adversarial.png")
    pert_img.save(f"{img_dir}/perturbation_vis.png")  # 可视化版本
    torch.save(perturbation.detach().cpu(), f"{img_dir}/perturbation.pt")  # 原始张量
    
    print(f"Saved adversarial images to {img_dir}/")


def attack(config: AttackConfig) -> None:
    """主攻击函数，使用配置对象管理参数"""
    _, test_dataset = config.datasets
    test_dataset = get_subset(dataset=test_dataset, frac=config.fraction)

    # 获得提示词映射
    img_id_to_train_prompt = get_img_id_train_prompt_map(config.prompt_num)
    # 获得环境前缀映射
    img_id_to_environment = get_img_id_environment_map()

    tpoch = tqdm(test_dataset)
    for id, item in enumerate(tpoch):
        # 加载文本提示
        img_id = item["image_id"]
        total_prompt_list = img_id_to_train_prompt[img_id]
        environment = img_id_to_environment[img_id]
        print(f"\n当前图像 {img_id} 对应的提示词数量: {len(total_prompt_list)}")

        # 初始化方法特定变量
        method_specific_vars = {}
        combined_embeddings = None
        
        # 初始化对抗文本嵌入
        if config.method in ["baseline", "multi_prompt", "embed_noise", "token_noise", "grad_embed_noise"]:
            print(f"This is {config.method} method")
        elif config.method == "embed_adv":
            print("This is embed_adv method")
            target_token_ids = config.processor.tokenizer.encode(config.target_text, add_special_tokens=False)
            adversarial_embeddings = config.eval_model.model.get_input_embeddings()(
                torch.tensor(target_token_ids, device=config.device)
            ).repeat(1, config.adversarial_length // len(target_token_ids) + 1, 1)[:, :config.adversarial_length, :]
            adversarial_embeddings = adversarial_embeddings.clone().detach().requires_grad_(True)
            # 保存初始嵌入
            adversarial_embeddings_init = adversarial_embeddings.clone().detach()
            
            method_specific_vars['adversarial_embeddings'] = adversarial_embeddings
            method_specific_vars['adversarial_embeddings_init'] = adversarial_embeddings_init  
        elif config.method == "token_adv":
            print("This is token_adv method")
            target_token_ids = config.processor.tokenizer.encode(config.target_text, add_special_tokens=False)
            # 初始化为目标token的浮点形式
            adversarial_token_ids = torch.tensor(
                target_token_ids * (config.adversarial_length // len(target_token_ids) + 1),
                device=config.device
            )[:config.adversarial_length].float().requires_grad_(True)
            # 保存初始值用于约束
            token_ids_init = adversarial_token_ids.clone().detach()
            
            method_specific_vars['adversarial_token_ids'] = adversarial_token_ids
            method_specific_vars['token_ids_init'] = token_ids_init
        else:
            raise ValueError(f"未知的对抗方法: {config.method}")

        # 初始化提示词轮换顺序
        item_images = [[item["image"]]]
        input_x_original = config.eval_model._prepare_images(item_images, normalize=False).to(config.device).requires_grad_(False)
        image_perturbation = torch.zeros_like(input_x_original, device=config.device).requires_grad_(True)
        best_loss = torch.tensor(float('inf'))
        best_attack = None

        # 先为每个提示词生成可优化的噪声
        total_text_noise = []
        if config.method == "grad_embed_noise":
            for i in range(len(total_prompt_list)):
                current_question = total_prompt_list[i]
                current_question, current_answer = config.eval_model.get_vqa_prompt(question=current_question, answer=config.target_text)
                # 处理问题部分
                question_inputs = config.processor(
                    text=[current_question],
                    padding=True,
                    truncation=True,
                    max_length=1000,
                    return_tensors="pt"
                ).to(config.device)
                question_embeddings = config.eval_model.model.get_input_embeddings()(question_inputs['input_ids'])
                # 生成可优化的噪声
                noise_scale = 0.3
                text_noise = torch.randn_like(question_embeddings) * noise_scale
                text_noise = text_noise.clone().detach().requires_grad_(True)
                total_text_noise.append(text_noise)
            
            method_specific_vars['total_text_noise'] = total_text_noise

        # 提示词轮换逻辑
        access_order = list(range(len(total_prompt_list)))
        random.shuffle(access_order)
        access_order = deque(access_order)
        index_count = 0

        for ep in range(config.iters):
            # 提示词轮换控制
            if index_count != 0 and index_count % len(total_prompt_list) == 0:
                rotation_offset = random.randint(0, len(total_prompt_list)-1)
                access_order.rotate(rotation_offset)
                index_count = 0
            text_idx = access_order[index_count]
            index_count += 1
            
            # 保存当前文本索引到方法特定变量中
            method_specific_vars['text_idx'] = text_idx

            # 获得当前提示词
            current_question = total_prompt_list[text_idx]
            if config.debug:
                print(f"[DEBUG] 当前提示词: {current_question}")

            # 问题处理
            if config.method == "baseline":
                # 空问题
                current_question = ""
            elif config.method in ["multi_prompt", "embed_noise", "token_noise", "grad_embed_noise"]:
                # 原始问题
                current_question = current_question
            elif config.method in ["embed_adv", "token_adv"]:
                # 添加了环境前缀的问题
                current_question = f"Against the background of {environment}, {current_question}"
            
            # 获得vqa模板问题
            current_question, current_answer = config.eval_model.get_vqa_prompt(question=current_question, answer=config.target_text)
            current_text = current_question + current_answer
            if config.debug:
                print(f"[DEBUG] 当前的vqa模板问题: {current_text}")
            
            # 获得整个文本的input_ids和attention_mask
            current_inputs = config.processor(
                text=[current_text],
                padding=True,
                truncation=True,
                max_length=1000,
                return_tensors="pt"
            ).to(config.device)
            
            # 获得问题部分的input_ids, attention_mask和embeddings
            question_inputs = config.processor(
                text=[current_question],
                padding=True,
                truncation=True,
                max_length=1000,
                return_tensors="pt"
            ).to(config.device)
            question_embeddings = config.eval_model.model.get_input_embeddings()(question_inputs['input_ids'])

            # 处理答案部分的input_ids, attention_mask和embeddings
            answer_inputs = config.processor.tokenizer.encode(
                current_answer,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(config.device).detach()
            answer_embeddings = config.eval_model.model.get_input_embeddings()(answer_inputs)

            # 获得目标文本的token_ids
            current_target = config.processor.tokenizer.encode(
                config.target_text,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(config.device).detach()

            # 根据方法构建输入和标签
            if config.method in ["baseline", "multi_prompt"]:
                current_inputs, labels = build_inputs_for_baseline_and_multi_prompt(current_inputs, current_target)
            elif config.method == "embed_noise":
                current_inputs, labels, combined_embeddings = build_inputs_for_embed_noise(
                    question_inputs, answer_inputs, question_embeddings, answer_embeddings,
                    current_target
                )
            elif config.method == "token_noise":
                current_inputs, labels = build_inputs_for_token_noise(
                    question_inputs, answer_inputs, current_target, config.processor
                )
            elif config.method == "grad_embed_noise":
                current_inputs, labels, combined_embeddings = build_inputs_for_grad_embed_noise(
                    question_inputs, answer_inputs, question_embeddings, answer_embeddings,
                    current_target, method_specific_vars['total_text_noise'][text_idx]
                )
            elif config.method == "embed_adv":
                current_inputs, labels, combined_embeddings = build_inputs_for_embed_adv(
                    question_inputs, answer_inputs, question_embeddings, answer_embeddings,
                    current_target, method_specific_vars['adversarial_embeddings'],
                    config.adversarial_length, config.processor
                )
            elif config.method == "token_adv":
                current_inputs, labels = build_inputs_for_token_adv(
                    question_inputs, answer_inputs, current_target,
                    method_specific_vars['adversarial_token_ids'], config.adversarial_length
                )
    
            # 对抗图像
            input_x = input_x_original + image_perturbation

            # 设置嵌入
            if config.method in ["embed_noise", "embed_adv", "grad_embed_noise"]:
                config.eval_model.set_custom_embeddings(combined_embeddings)

            # 前向传播计算梯度
            outputs = config.eval_model.model(
                input_ids=current_inputs['input_ids'],
                pixel_values=input_x,
                attention_mask=current_inputs['attention_mask'],
                labels=labels
            )

            # 清除嵌入
            if config.method in ["embed_noise", "embed_adv", "grad_embed_noise"]:
                config.eval_model.clear_custom_embeddings()

            loss = outputs.loss
            
            # 更新扰动参数
            best_loss, best_attack, method_specific_vars = update_perturbations(
                config, loss, image_perturbation, best_loss, best_attack,
                method_specific_vars, ep
            )

        # 保存对抗攻击结果
        save_adversarial_results(config, input_x_original, image_perturbation, item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对抗攻击参数配置')
    
    parser.add_argument('--model_name', type=str, default='blip2', help='模型名称')
    parser.add_argument('--method', type=str, default='token_adv', 
                       choices=['baseline', 'multi_prompt', 'embed_noise', 'token_noise', 'grad_embed_noise', 'embed_adv', 'token_adv'],
                       help='对抗方法')
    parser.add_argument('--prompt_num', type=int, default=50, help='提示词数量')
    parser.add_argument('--adversarial_length', type=int, default=10, help='对抗文本长度')
    parser.add_argument('--device', type=int, default=0, help='GPU设备号')
    parser.add_argument('--iters', type=int, default=800, help='迭代次数')
    parser.add_argument('--fraction', type=float, default=0.05, help='数据集采样比例')
    parser.add_argument('--epsilon', type=float, default=32/255, help='扰动大小限制')
    parser.add_argument('--alpha', type=float, default=1/255, help='学习率')
    parser.add_argument('--debug', action='store_true', help='启用调试信息')
    
    args = parser.parse_args()
    
    # 加载模型和数据集
    module = importlib.import_module(f"models.my{args.model_name}")
    eval_model = load_model(args.device, module, args.model_name)
    train_dataset, test_dataset = load_dataset()

    # 创建配置对象
    target_text = "Unknown"
    config = AttackConfig(
        method=args.method,
        target_text=target_text,
        eval_model=eval_model,
        adversarial_length=args.adversarial_length,
        datasets=(train_dataset, test_dataset),
        fraction=args.fraction,
        prompt_num=args.prompt_num,
        iters=args.iters,
        device=args.device,
        epsilon=args.epsilon,
        alpha=args.alpha,
        debug=args.debug
    )
    
    # 执行攻击
    attack(config)
    