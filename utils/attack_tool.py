import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import numpy as np
import json
import torch
from eval_datasets import COCO_Dataset

MODEL_CONFIG = {
    "blip2": {
        "lm_path": "models/Salesforce/blip2-opt-2.7b",
        "processor_path": "models/Salesforce/blip2-opt-2.7b"
    }
}

DATASET_CONFIG = {
    "vqav2": {
        "train": {
            "image_dir": "data/train2014",
            "questions_json": "data/v2_OpenEnded_mscoco_train2014_questions.json",
            "annotations_json": "data/v2_mscoco_train2014_annotations.json"
        },
        "test": {
            "image_dir": "data/val2014",
            "questions_json": "data/filtered_v2_OpenEnded_mscoco_val2014_questions.json",
            "annotations_json": "data/filtered_v2_mscoco_val2014_annotations.json"
        }
    }
}

class RoundWithSTE(torch.autograd.Function):
    """
    直通估计器 (Straight-Through Estimator, STE)
    允许在离散取整操作中传递梯度。
    
    用法：
        rounded_tokens = RoundWithSTE.apply(adversarial_token_ids)
    """
    @staticmethod
    def forward(ctx, input):
        # 前向传播：四舍五入并转为整数
        return input.round().long()
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：直接传递梯度（绕过取整操作的不可导性）
        return grad_output.clone()

def load_model(device, module, model_name):
    print("model_name is:", model_name)
    if model_name == "blip2":
        return load_blip_model(device, module)
    else:
        raise ValueError("model name is not valid")
    

def load_blip_model(device, module):
    model_args = {
        **MODEL_CONFIG["blip2"],  # 展开配置参数
        'device': device  # 单独添加设备参数
    }
    eval_model = module.EvalModel(model_args)
    return eval_model


def load_dataset(dataset_name="vqav2"):
    config = DATASET_CONFIG[dataset_name]
    
    train_dataset = COCO_Dataset(
        image_dir_path=config["train"]["image_dir"],
        question_path=config["train"]["questions_json"],
        annotations_path=config["train"]["annotations_json"],
        is_train=True
    )

    test_dataset = COCO_Dataset(
        image_dir_path=config["test"]["image_dir"],
        question_path=config["test"]["questions_json"],
        annotations_path=config["test"]["annotations_json"],
        is_train=False
    )
    return train_dataset, test_dataset


def get_subset(dataset, frac):
    if frac < 1.0:
        dataset_size = len(dataset)
        subset_size = int(frac * dataset_size)
        indices = np.arange(subset_size)
        dataset = torch.utils.data.Subset(dataset, indices)                
    return dataset


def get_intended_token_ids(input_ids, target_id):
    padding = torch.full_like(input_ids, -100)
    
    if target_id.dim() == 2:
        target = target_id.squeeze(0)
    else:
        target = target_id
    
    padding[:, -len(target):] = target
    return padding


def get_img_id_train_prompt_map(num):
    path = f"data/prompt_list/num_{num}/id_to_question.json"
    with open(path, 'r') as f:
        id_ques_map = json.load(f)
    return id_ques_map


def get_img_id_environment_map():
    path = f"data/id_to_environment.json"
    with open(path, 'r') as f:
        id_env_map = json.load(f)
    return id_env_map