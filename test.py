import json
import os
import re
import more_itertools
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import torch
from PIL import Image
from dataclasses import dataclass
import importlib
import argparse
from utils.attack_tool import (
    load_model
)

from utils.eval_tool import (
    vqa_agnostic_instruction,
    load_img_specific_questions,
    cls_instruction,
    cap_instruction
)

@dataclass
class EvalConfig:
    """评估配置类，用于统一管理评估参数"""
    model_name: str = "blip2"
    method: str = "my_embedding"
    device: int = 0
    target_text: str = "I don't know."
    eval_batch_size: int = 4
    max_generation_length: int = 50
    num_beams: int = 3
    length_penalty: float = -1.0
    
    @property
    def device_str(self) -> str:
        """返回设备字符串"""
        return f"cuda:{self.device}" if self.device >= 0 else "cpu"

class AttackEvaluator:
    def __init__(self, eval_model, config: EvalConfig):
        """
        初始化评估器
        :param eval_model: 已加载的模型对象
        :param config: 评估配置对象
        """
        self.eval_model = eval_model
        self.config = config
        self.device = config.device_str
        
        # 任务配置
        self.task_list = ["vqa", "vqa_specific", "cls", "cap"]
        self.task_prompts = None
        self._load_prompts()
        
    def _load_prompts(self):
        """加载所有任务的提示词模板"""
        self.task_prompts = {
            "vqa": vqa_agnostic_instruction(),
            "vqa_specific": load_img_specific_questions(),
            "cls": cls_instruction(),
            "cap": cap_instruction()
        }

    def evaluate(
        self,
        attack: torch.Tensor,
        img_id: str,
        test_images: List,
        target_text: str
    ) -> Dict[str, Any]:
        """
        执行完整评估流程
        :param attack: 对抗扰动张量
        :param img_id: 图片ID
        :param test_images: 测试图片列表
        :param target_text: 目标文本
        :return: 包含所有评估结果的字典
        """
        results = {
            "img_id": img_id,
            "target": target_text,
            "tasks": {}
        }
        
        # 初始化VQA统计
        vqa_stats = self._init_vqa_stats()
        
        # 评估每个任务
        for task_name in self.task_list:
            # debug信息
            print(task_name)

            task_results = self._evaluate_task(
                task_name=task_name,
                attack=attack,
                img_id=img_id,
                test_images=test_images,
                target_text=target_text,
                vqa_stats=vqa_stats
            )
            results["tasks"][task_name] = task_results
        
        results["vqa_stats"] = vqa_stats
        return results

    def _evaluate_task(
        self,
        task_name: str,
        attack: torch.Tensor,
        img_id: str,
        test_images: List,
        target_text: str,
        vqa_stats: Dict
    ) -> Dict:
        """评估单个任务"""
        task_results = {
            "success_count": 0,
            "target_success_count": 0,
            "total": 0
        }
        
        instructions = self._get_task_instructions(task_name, img_id)
        
        # 分批处理
        for batch in more_itertools.chunked(instructions, self.config.eval_batch_size):
            # 生成对抗样本输出
            outputs = self._generate_outputs(
                task_name=task_name,
                attack=attack,
                test_images=test_images,
                batch_instructions=batch
            )
            
            # 生成干净样本输出
            clean_outputs = self._generate_outputs(
                task_name=task_name,
                attack=None,
                test_images=test_images,
                batch_instructions=batch
            )
            
            # 处理输出结果
            processed_outputs = [self._postprocess(p) for p in outputs]
            processed_clean = [self._postprocess(p) for p in clean_outputs]
            
            # 统计结果
            self._update_task_stats(
                task_results, processed_outputs, processed_clean, 
                batch, task_name, target_text, vqa_stats
            )
        
        return task_results
        
    def _get_task_instructions(self, task_name: str, img_id: str) -> List[str]:
        """获取特定任务的指令列表"""
        instructions = self.task_prompts[task_name]
        if task_name == "vqa_specific":
            instructions = instructions[img_id][:10]  # 取前10个特定问题
        return instructions
        
    def _update_task_stats(
        self,
        task_results: Dict,
        processed_outputs: List[str],
        processed_clean: List[str],
        batch: List[str],
        task_name: str,
        target_text: str,
        vqa_stats: Dict
    ) -> None:
        """更新任务统计信息"""
        target_text_lower = target_text.lower().split("<")[0].strip()
        
        for i, pred in enumerate(processed_outputs):
            task_results["total"] += 1
            pred_lower = pred.strip().lower()
            
            # 目标攻击成功判断
            if pred_lower == target_text_lower:
                task_results["target_success_count"] += 1
            
            # 非目标攻击成功判断
            if pred != processed_clean[i]:
                task_results["success_count"] += 1
            
            # VQA特定统计
            if task_name in ["vqa", "vqa_specific"]:
                q_type = self._get_vqa_type(batch[i])
                if pred_lower == target_text_lower:
                    vqa_stats[q_type]["success"] += 1
                vqa_stats[q_type]["total"] += 1

    def _generate_outputs(
        self,
        task_name: str,
        attack: Optional[torch.Tensor],
        test_images: List,
        batch_instructions: List[str]
    ) -> List[str]:
        """生成模型输出(统一处理对抗/干净样本)"""
        # 构建输入文本
        texts = [self.eval_model.get_vqa_prompt(q) for q in batch_instructions]
        
        # 准备批量图像
        batch_images = test_images * len(batch_instructions)
        
        # 生成输出
        if attack is not None:
            return self.eval_model.get_outputs_attack(
                attack=attack,
                batch_images=batch_images,
                batch_text=texts,
                max_generation_length=self.config.max_generation_length,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty
            )
        else:
            return self.eval_model.get_outputs(
                batch_images=batch_images,
                batch_text=texts,
                max_generation_length=self.config.max_generation_length,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty
            )

    @staticmethod
    def _postprocess(prediction: str) -> str:
        """后处理模型输出"""
        answer_match = re.search(r"Answer:(.*?)(?:Question|$)", prediction, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            # 进一步处理可能的分隔符
            return re.split(r", |\.\s", answer, 1)[0]
        return prediction

    @staticmethod
    def _init_vqa_stats() -> Dict:
        """初始化VQA统计字典"""
        return {
            "number": {"success": 0, "total": 0},
            "yes_no": {"success": 0, "total": 0},
            "what": {"success": 0, "total": 0},
            "where": {"success": 0, "total": 0},
            "other": {"success": 0, "total": 0}
        }
    
    @staticmethod
    def _get_vqa_type(question: str) -> str:
        """
        静态方法：根据问题文本判断VQA问题类型
        :param question: 输入的问题文本
        :return: 问题类型标签 ("number"/"where"/"what"/"yes_no"/"other")
        """
        question_lower = question.lower()
        if question_lower.startswith("how many"):
            return "number"
        elif question_lower.startswith("where"):
            return "where"
        elif question_lower.startswith("what"):
            return "what"
        elif question_lower.startswith(("is", "are", "will", "can", "do", "does", 
                                    "has", "have", "did", "were", "was", 
                                    "should", "any")):
            return "yes_no"
        else:
            return "other"

    def save_results(self, results: Dict, output_dir: str, iteration: int = None):
        """保存评估结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"eval_results_{iteration}.json" if iteration else "eval_results.json"
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(results, f, indent=2)


def evaluate_method_performance(eval_model, config: EvalConfig, method_name: str) -> Dict[str, Any]:
    """
    评测指定方法生成的对抗样本
    :param eval_model: 已加载的评估模型
    :param config: 评估配置对象
    :param method_name: 要评测的方法名称
    :return: 评测结果汇总
    """
    # 初始化评估器
    evaluator = AttackEvaluator(eval_model, config)
    
    # 结果统计字典
    method_results = {
        "total_images": 0,
        "task_results": defaultdict(lambda: {
            "success_rate": 0.0,
            "target_success_rate": 0.0,
            "total": 0
        }),
        "vqa_stats": defaultdict(lambda: {
            "success": 0,
            "total": 0
        })
    }

    # 准备评估目录
    method_dir = get_adversarial_images_dir(method_name)
    
    # 获取所有图像ID子目录
    img_dirs = [d for d in os.listdir(method_dir) if os.path.isdir(os.path.join(method_dir, d))]
    
    for img_id in tqdm(img_dirs, desc=f"Evaluating {method_name}"):
        # 加载样本数据
        attack_data = load_attack_data(img_dir=os.path.join(method_dir, img_id), 
                                      device=config.device_str, 
                                      eval_model=eval_model)
        
        # 执行评估
        results = evaluator.evaluate(
            attack=attack_data["perturbation"],
            img_id=img_id,
            test_images=[attack_data["original_image"]],
            target_text=config.target_text
        )
        
        # 汇总结果
        update_method_results(method_results, results)
    
    # 计算平均成功率
    calculate_average_success_rates(method_results)
    
    # 保存结果
    output_dir = save_evaluation_results(method_results, method_name, evaluator)
    
    print(f"\n{method_name} 评测完成！结果已保存到 {output_dir}/")
    return method_results

def get_adversarial_images_dir(method_name: str) -> str:
    """
    获取对抗图像目录
    :param method_name: 方法名称
    :return: 目录路径
    :raises FileNotFoundError: 如果目录不存在
    """
    method_dir = os.path.join("adversarial_images", method_name)
    if not os.path.exists(method_dir):
        raise FileNotFoundError(f"找不到方法目录: {method_dir}")
    return method_dir

def load_attack_data(img_dir: str, device: str, eval_model) -> Dict[str, Any]:
    """
    加载攻击数据
    :param img_dir: 图像目录
    :param device: 设备字符串
    :param eval_model: 评估模型
    :return: 包含扰动和原始图像的字典
    """
    # 加载对抗扰动
    pert_path = os.path.join(img_dir, "perturbation.pt")
    perturbation = torch.load(pert_path).to(device)
    
    # 加载原始图像
    orig_img = Image.open(os.path.join(img_dir, "original.png"))
    orig_img = eval_model._prepare_images([[orig_img]], normalize=False)
    
    return {
        "perturbation": perturbation,
        "original_image": orig_img
    }

def update_method_results(method_results: Dict, results: Dict) -> None:
    """
    更新方法评估结果
    :param method_results: 方法评估结果字典
    :param results: 当前图像的评估结果
    """
    method_results["total_images"] += 1
    
    # 汇总任务结果
    for task_name, task_res in results["tasks"].items():
        method_results["task_results"][task_name]["success_rate"] += task_res["success_count"] / task_res["total"]
        method_results["task_results"][task_name]["target_success_rate"] += task_res["target_success_count"] / task_res["total"]
        method_results["task_results"][task_name]["total"] += task_res["total"]
    
    # 汇总VQA统计
    for q_type, stats in results["vqa_stats"].items():
        method_results["vqa_stats"][q_type]["success"] += stats["success"]
        method_results["vqa_stats"][q_type]["total"] += stats["total"]

def calculate_average_success_rates(method_results: Dict) -> None:
    """
    计算平均成功率
    :param method_results: 方法评估结果字典
    """
    total_images = method_results["total_images"]
    if total_images > 0:
        for task in method_results["task_results"]:
            method_results["task_results"][task]["success_rate"] /= total_images
            method_results["task_results"][task]["target_success_rate"] /= total_images

def save_evaluation_results(method_results: Dict, method_name: str, evaluator: AttackEvaluator) -> str:
    """
    保存评估结果
    :param method_results: 方法评估结果字典
    :param method_name: 方法名称
    :param evaluator: 评估器实例
    :return: 输出目录路径
    """
    output_dir = os.path.join("evaluation_results", method_name)
    os.makedirs(output_dir, exist_ok=True)
    evaluator.save_results(method_results, output_dir)
    return output_dir


def create_config_from_args(args) -> EvalConfig:
    """
    从命令行参数创建配置对象
    :param args: 命令行参数
    :return: 评估配置对象
    """
    return EvalConfig(
        model_name=args.model_name,
        method=args.method,
        device=args.device,
        target_text=args.target_text,
        eval_batch_size=args.eval_batch_size,
        max_generation_length=args.max_generation_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty
    )

def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """
    打印评估结果摘要
    :param results: 评估结果字典
    """
    print("\n评估结果摘要:")
    print(f"评估样本总数: {results['total_images']}")
    print("{:<15} {:<15} {:<15}".format(
        "Task", "Success Rate", "Target Success"))
    
    for task, res in results["task_results"].items():
        print("{:<15} {:<15.1f}% {:<15.1f}%".format(
            task, 
            res["success_rate"] * 100,
            res["target_success_rate"] * 100))

def main():
    # 1. 配置单个方法的评估参数
    parser = argparse.ArgumentParser(description='对抗攻击评估参数配置')
    
    parser.add_argument('--model_name', type=str, default='blip2', help='使用的模型名称')
    parser.add_argument('--method', type=str, default='my_embedding', 
                       help='要评估的方法名称')
    parser.add_argument('--device', type=int, default=0, help='GPU设备号')
    parser.add_argument('--target_text', type=str, default='I don\'t know.', 
                       help='目标攻击文本')
    parser.add_argument('--eval_batch_size', type=int, default=4, 
                       help='评估批大小')
    parser.add_argument('--max_generation_length', type=int, default=50, 
                       help='生成文本最大长度')
    parser.add_argument('--num_beams', type=int, default=3, 
                       help='beam search参数')
    parser.add_argument('--length_penalty', type=float, default=-1.0, 
                       help='生成长度惩罚')
    
    args = parser.parse_args()
    
    # 创建配置对象
    config = create_config_from_args(args)

    # 2. 加载模型
    print(f"正在加载 {config.model_name} 模型...")
    module = importlib.import_module(f"models.my{config.model_name}")
    eval_model = load_model(config.device, module, config.model_name)
    
    # 3. 执行单个方法评估
    print(f"\n开始评估方法: {config.method}")
    results = evaluate_method_performance(
        eval_model=eval_model,
        config=config,
        method_name=config.method
    )
    
    # 4. 打印关键结果
    print_evaluation_summary(results)

if __name__ == "__main__":    
    main()