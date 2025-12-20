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


def attack(config: AttackConfig) -> None:
    """使用embed_adv方法进行对抗攻击"""
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

        # 初始化提示词轮换顺序
        item_images = [[item["image"]]]
        input_x_original = config.eval_model._prepare_images(item_images, normalize=False).to(config.device).requires_grad_(False)
        image_perturbation = torch.zeros_like(input_x_original, device=config.device).requires_grad_(True)
        best_loss = torch.tensor(float('inf'))
        best_attack = None

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
            
            # 处理问题部分
            question_inputs = config.processor(
                text=[current_question],
                padding=True,
                truncation=True,
                max_length=1000,
                return_tensors="pt"
            ).to(config.device)
            question_embeddings = config.eval_model.model.get_input_embeddings()(question_inputs['input_ids'])

            # 处理答案部分的input_ids和embeddings
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

            # 构建输入和标签（原build_inputs_for_embed_adv函数的功能）
            # 拼接embeddings
            combined_embeddings = torch.cat([
                question_embeddings,
                method_specific_vars['adversarial_embeddings'],
                answer_embeddings
            ], dim=1)

            # 构建input_ids和attention_mask
            pad_token_id = config.processor.tokenizer.pad_token_id
            padded_input_ids = torch.cat([
                question_inputs.input_ids,
                torch.full((1, config.adversarial_length), pad_token_id, device=question_inputs.input_ids.device),
                answer_inputs
            ], dim=1)
            combined_attention_mask = torch.cat([
                question_inputs.attention_mask,
                torch.ones((1, config.adversarial_length), device=question_inputs.input_ids.device),
                torch.ones_like(answer_inputs)
            ], dim=1)
            current_inputs = {
                'input_ids': padded_input_ids,
                'attention_mask': combined_attention_mask
            }

            # 处理labels
            labels = get_intended_token_ids(current_inputs['input_ids'], current_target)

            # 对抗图像
            input_x = input_x_original + image_perturbation

            # 设置嵌入
            config.eval_model.set_custom_embeddings(combined_embeddings)

            # 前向传播计算梯度
            outputs = config.eval_model.model(
                input_ids=current_inputs['input_ids'],
                pixel_values=input_x,
                attention_mask=current_inputs['attention_mask'],
                labels=labels
            )

            # 清除嵌入
            config.eval_model.clear_custom_embeddings()

            loss = outputs.loss
            
            # 更新扰动参数（原update_perturbations_embed_adv函数的功能）
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

            # 更新文本扰动
            grad_embeddings = method_specific_vars['adversarial_embeddings'].grad.detach()
            update = config.alpha * torch.sign(grad_embeddings) * (1 - ep/config.iters)
            method_specific_vars['adversarial_embeddings'].data = torch.clamp(
                method_specific_vars['adversarial_embeddings'].data + update,
                min=method_specific_vars['adversarial_embeddings_init'] - 1,  # 语义保持约束
                max=method_specific_vars['adversarial_embeddings_init'] + 1
            )
            method_specific_vars['adversarial_embeddings'].grad.zero_()

        # 保存对抗攻击结果（原save_adversarial_results函数的功能）
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对抗攻击参数配置')
    
    parser.add_argument('--model_name', type=str, default='blip2', help='模型名称')
    parser.add_argument('--method', type=str, default='embed_adv', 
                       choices=['embed_adv'],
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
    module = importlib.import_module(f"models.{args.model_name}")
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