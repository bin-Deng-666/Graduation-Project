import os
import random
from argparse import Namespace
from typing import Tuple
from collections import deque

import importlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse

from models import BaseEvalModel
from utils.my_attack_tool import (
    load_model,
    load_dataset,
    get_subset,
    get_intended_token_ids,
    get_img_id_train_prompt_map,
    get_img_id_environment_map,
    RoundWithSTE
)

def attack(
        method: str,
        target_text: str,
        adversarial_length: int,
        eval_model: BaseEvalModel,
        datasets: Tuple[Dataset, Dataset],
        fraction: float,
        prompt_num: int,
        iters: int,
        device: int,
        epsilon: float=32/255,
        alpha: float=1/255,
        debug=False
) -> None:
    _, test_dataset = datasets
    test_dataset = get_subset(dataset=test_dataset, frac=fraction)
    processor = eval_model.processor

    # 获得提示词映射
    img_id_to_train_prompt = get_img_id_train_prompt_map(prompt_num)
    # 获得环境前缀映射
    img_id_to_environment = get_img_id_environment_map()

    tpoch = tqdm(test_dataset)
    for id, item in enumerate(tpoch):
        # 加载文本提示
        img_id = item["image_id"]
        total_prompt_list = img_id_to_train_prompt[img_id]
        environment = img_id_to_environment[img_id]
        print(f"\n当前图像 {img_id} 对应的提示词数量: {len(total_prompt_list)}")

        # 初始化对抗文本嵌入
        if method in ["baseline", "multi_prompt", "embed_noise", "token_noise", "grad_embed_noise"]:
            print(f"This is {method} method")
        elif method == "embed_adv":
            print("This is embed_adv method")
            target_token_ids = processor.tokenizer.encode(target_text, add_special_tokens=False)
            adversarial_embeddings = eval_model.model.get_input_embeddings()(
                torch.tensor(target_token_ids, device=device)
            ).repeat(1, adversarial_length // len(target_token_ids) + 1, 1)[:, :adversarial_length, :]
            adversarial_embeddings = adversarial_embeddings.clone().detach().requires_grad_(True)
            # 保存初始嵌入
            adversarial_embeddings_init = adversarial_embeddings.clone().detach()
        elif method == "token_adv":
            print("This is token_adv method")
            target_token_ids = processor.tokenizer.encode(target_text, add_special_tokens=False)
            # 初始化为目标token的浮点形式
            adversarial_token_ids = torch.tensor(
                target_token_ids * (adversarial_length // len(target_token_ids) + 1),
                device=device
            )[:adversarial_length].float().requires_grad_(True)
            # 保存初始值用于约束
            token_ids_init = adversarial_token_ids.clone().detach()
        else:
            raise ValueError(f"未知的对抗方法: {method}")

        # 初始化提示词轮换顺序
        item_images = [[item["image"]]]
        input_x_original = eval_model._prepare_images(item_images, normalize=False).to(device).requires_grad_(False)
        image_perturbation = torch.zeros_like(input_x_original, device=device).requires_grad_(True)
        best_loss = torch.tensor(float('inf'))
        best_attack = None

        # 先为每个提示词生成可优化的噪声
        total_text_noise = []
        if method == "grad_embed_noise":
            for i in range(len(total_prompt_list)):
                current_question = total_prompt_list[i]
                current_question, current_answer = eval_model.get_vqa_prompt(question=current_question, answer=target_text)
                # 处理问题部分
                question_inputs = processor(
                    text=[current_question],
                    padding=True,
                    truncation=True,
                    max_length=1000,
                    return_tensors="pt"
                ).to(device)
                question_embeddings = eval_model.model.get_input_embeddings()(question_inputs['input_ids'])
                # 生成可优化的噪声
                noise_scale = 0.3
                text_noise = torch.randn_like(question_embeddings) * noise_scale
                text_noise = text_noise.clone().detach().requires_grad_(True)
                total_text_noise.append(text_noise)

        # 提示词轮换逻辑
        access_order = list(range(len(total_prompt_list)))
        random.shuffle(access_order)
        access_order = deque(access_order)
        index_count = 0

        for ep in range(iters):
            # 提示词轮换控制
            if index_count != 0 and index_count % len(total_prompt_list) == 0:
                rotation_offset = random.randint(0, len(total_prompt_list)-1)
                access_order.rotate(rotation_offset)
                index_count = 0
            text_idx = access_order[index_count]
            index_count += 1

            # 获得当前提示词
            current_question = total_prompt_list[text_idx]
            if debug:
                print(f"[DEBUG] 当前提示词: {current_question}")

            # 问题处理
            if method == "baseline":
                # 空问题
                current_question = ""
            elif method in ["multi_prompt", "embed_noise", "token_noise", "grad_embed_noise"]:
                # 原始问题
                current_question = current_question
            elif method in ["embed_adv", "token_adv"]:
                # 添加了环境前缀的问题
                current_question = f"Against the background of {environment}, {current_question}"
            # 获得vqa模板问题
            current_question, current_answer = eval_model.get_vqa_prompt(question=current_question, answer=target_text)
            current_text = current_question + current_answer
            if debug:
                print(f"[DEBUG] 当前的vqa模板问题: {current_text}")
            
            # 获得整个文本的input_ids和attention_mask
            current_inputs = processor(
                text=[current_text],
                padding=True,
                truncation=True,
                max_length=1000,
                return_tensors="pt"
            ).to(device)
            
            # 获得问题部分的input_ids, attention_mask和embeddings
            question_inputs = processor(
                text=[current_question],
                padding=True,
                truncation=True,
                max_length=1000,
                return_tensors="pt"
            ).to(device)
            question_embeddings = eval_model.model.get_input_embeddings()(question_inputs['input_ids'])

            # 处理答案部分的input_ids, attention_mask和embeddings
            answer_inputs = processor.tokenizer.encode(
                current_answer,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(device).detach()
            answer_embeddings = eval_model.model.get_input_embeddings()(answer_inputs)

            # 获得目标文本的token_ids
            current_target = processor.tokenizer.encode(
                target_text,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(device).detach()

            if method in ["baseline", "multi_prompt"]:
                # 构建input_ids和attention_mask
                current_inputs = {
                    'input_ids': current_inputs.input_ids,
                    'attention_mask': current_inputs.attention_mask
                }

                # 处理labels
                labels = get_intended_token_ids(current_inputs['input_ids'], current_target)
            elif method == "embed_noise":
                # 添加高斯噪声
                noise_scale = 0.3
                noisy_question_embeddings = question_embeddings + torch.randn_like(question_embeddings) * noise_scale

                # 拼接带噪声的问题嵌入和原始答案嵌入
                combined_embeddings = torch.cat([
                    noisy_question_embeddings,
                    answer_embeddings
                ], dim=1)

                # 构建input_ids和attention_mask
                combined_input_ids = torch.cat([
                    question_inputs.input_ids,
                    answer_inputs
                ], dim=1) 
                combined_attention_mask = torch.cat([
                    question_inputs.attention_mask,
                    torch.ones_like(answer_inputs)
                ], dim=1)
                current_inputs = {
                    'input_ids': combined_input_ids,
                    'attention_mask': combined_attention_mask
                }

                # 处理labels
                labels = get_intended_token_ids(current_inputs['input_ids'], current_target)
            elif method == "token_noise":
                # 对input_ids添加整数噪声
                noisy_input_ids = question_inputs.input_ids.clone()
                # 扰动参数
                perturb_prob = 0.3  # 30%的概率扰动每个token
                max_perturb = 2     # 最大扰动幅度±2
                # 生成扰动掩码和扰动值
                perturb_mask = torch.rand_like(noisy_input_ids.float()) < perturb_prob
                perturb_amount = torch.randint(-max_perturb, max_perturb+1, noisy_input_ids.shape, device=device)
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
                current_inputs = {
                    'input_ids': combined_input_ids,
                    'attention_mask': combined_attention_mask
                }

                # 处理labels
                labels = get_intended_token_ids(current_inputs['input_ids'], current_target)
            elif method == "grad_embed_noise":
                # 使用预先生成的可优化噪声
                text_noise = total_text_noise[text_idx]
                noisy_question_embeddings = question_embeddings + text_noise
                
                # 拼接带噪声的问题嵌入和原始答案嵌入
                combined_embeddings = torch.cat([
                    noisy_question_embeddings,
                    answer_embeddings
                ], dim=1)

                # 构建input_ids和attention_mask
                combined_input_ids = torch.cat([
                    question_inputs.input_ids,
                    answer_inputs
                ], dim=1) 
                combined_attention_mask = torch.cat([
                    question_inputs.attention_mask,
                    torch.ones_like(answer_inputs)
                ], dim=1)
                current_inputs = {
                    'input_ids': combined_input_ids,
                    'attention_mask': combined_attention_mask
                }

                # 处理labels
                labels = get_intended_token_ids(current_inputs['input_ids'], current_target)
            elif method == "embed_adv":
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
                    torch.full((1, adversarial_length), pad_token_id, device=device),
                    answer_inputs
                ], dim=1)
                combined_attention_mask = torch.cat([
                    question_inputs.attention_mask,
                    torch.ones((1, adversarial_length), device=device),
                    torch.ones_like(answer_inputs)
                ], dim=1)
                current_inputs = {
                    'input_ids': padded_input_ids,
                    'attention_mask': combined_attention_mask
                }

                # 处理labels
                labels = get_intended_token_ids(current_inputs['input_ids'], current_target)
            elif method == "token_adv":
                # 构建input_ids和attention_mask
                rounded_token_ids = RoundWithSTE.apply(adversarial_token_ids)
                combined_token_ids = torch.cat([
                    question_inputs.input_ids,
                    rounded_token_ids.unsqueeze(0),  # 对抗token
                    answer_inputs
                ], dim=1)
                combined_attention_mask = torch.cat([
                    question_inputs.attention_mask,
                    torch.ones((1, adversarial_length), device=device),
                    torch.ones_like(answer_inputs)
                ], dim=1)
                current_inputs = {
                    'input_ids': combined_token_ids,
                    'attention_mask': combined_attention_mask
                }

                # 处理labels
                labels = get_intended_token_ids(current_inputs['input_ids'], current_target)
    
            # 对抗图像
            input_x = input_x_original + image_perturbation

            # 设置嵌入
            if method in ["embed_noise", "embed_adv", "grad_embed_noise"]:
                eval_model.set_custom_embeddings(combined_embeddings)

            # 前向传播计算梯度
            outputs = eval_model.model(
                input_ids=current_inputs['input_ids'],
                pixel_values=input_x,
                attention_mask=current_inputs['attention_mask'],
                labels=labels
            )

            # 清除嵌入
            if method in ["embed_noise", "embed_adv", "grad_embed_noise"]:
                eval_model.clear_custom_embeddings()

            loss = outputs.loss
            continue
            # 反向传播
            loss.backward()
            if loss < best_loss:
                best_loss = loss
                best_attack = image_perturbation.clone().detach()

            # 更新图像扰动
            grad_img = image_perturbation.grad.detach()
            image_perturbation.data = torch.clamp(
                image_perturbation.data - alpha * torch.sign(grad_img),
                min=-epsilon, # 像素值约束
                max=epsilon  
            )
            # 梯度清零
            image_perturbation.grad.zero_()

            if method == "embed_adv":
                # 更新文本扰动
                grad_embeddings = adversarial_embeddings.grad.detach()
                update = alpha * torch.sign(grad_embeddings) * (1 - ep/iters)
                adversarial_embeddings.data = torch.clamp(
                    adversarial_embeddings.data + update,
                    min=adversarial_embeddings_init - 1,  # 语义保持约束
                    max=adversarial_embeddings_init + 1
                )
                adversarial_embeddings.grad.zero_()
            elif method == "token_adv":
                 # 更新token_ids
                grad_token_ids = adversarial_token_ids.grad.detach()
                # 更新 token_ids（一步梯度更新 + 约束）
                adversarial_token_ids.data = token_ids_init + torch.clamp(
                    adversarial_token_ids.data - token_ids_init + alpha * grad_token_ids.sign(),
                    min=-2.0,  # 单步最大变化幅度
                    max=2.0
                )
                # 确保 token_ids 在有效范围内（0 到 vocab_size-1）
                vocab_size = processor.tokenizer.vocab_size
                adversarial_token_ids.data = torch.clamp(
                    adversarial_token_ids.data,
                    min=0,
                    max=vocab_size - 1
                )
                adversarial_token_ids.grad.zero_()
            elif method == "grad_embed_noise":
                # 更新文本噪声
                grad_noise = text_noise.grad.detach()
                text_noise.data = torch.clamp(
                    text_noise.data + alpha * grad_noise.sign() * (1 - ep/iters),
                    min=-0.5,
                    max=0.5
                )
                # 将更新后的噪声写回列表
                total_text_noise[text_idx] = text_noise
                text_noise.grad.zero_()

        # 创建保存目录
        output_dir = os.path.join("adversarial_images", f"{method}_p{prompt_num}")
        os.makedirs(output_dir, exist_ok=True)

        # 生成对抗图像和扰动
        # adversarial_image = input_x_original + best_attack
        # adversarial_image = adversarial_image.squeeze(0)
        # perturbation = best_attack.squeeze(0)
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
    parser.add_argument('--method', type=str, default='token_adv', 
                       choices=['baseline', 'multi_prompt', 'embed_noise', 'token_noise', 'grad_embed_noise', 'embed_adv', 'token_adv'],
                       help='对抗方法')
    parser.add_argument('--prompt_num', type=int, default=50, help='提示词数量')
    parser.add_argument('--adversarial_length', type=int, default=10, help='对抗文本长度')
    parser.add_argument('--device', type=int, default=0, help='GPU设备号')
    parser.add_argument('--iters', type=int, default=800, help='迭代次数')
    parser.add_argument('--fraction', type=float, default=0.05, help='数据集采样比例')
    
    args = parser.parse_args()
    
    module = importlib.import_module(f"models.my{args.model_name}")
    device = f"cuda:{args.device}"
    eval_model = load_model(args.device, module, args.model_name)
    train_dataset, test_dataset = load_dataset()

    target_text = "Unknown"
    attack(
        method=args.method, 
        target_text=target_text,
        eval_model=eval_model,
        adversarial_length=args.adversarial_length,
        datasets=(train_dataset, test_dataset),
        fraction=args.fraction,
        prompt_num=args.prompt_num,
        iters=args.iters,
        device=args.device
    )
    