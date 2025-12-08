# 标准库导入
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from typing import List

# 第三方库导入
import torch
from PIL import Image
from torchvision import transforms
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration
)
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

# 本地模块导入
from BaseEvalModel import BaseEvalModel

class EvalModel(BaseEvalModel):
    """My customized BLIP-2 model implementation."""
    
    def __init__(self, model_args):
        assert all(k in model_args for k in ["processor_path", "lm_path", "device"]), \
            "Missing required model arguments"
        
        self.device = model_args["device"] if model_args["device"] >= 0 else "cpu"
        
        # Initialize processor and model
        self.processor = Blip2Processor.from_pretrained(model_args["processor_path"])
        # self.model = Blip2ForConditionalGeneration.from_pretrained(model_args["lm_path"], torch_dtype=torch.float16)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_args["lm_path"])
        self.tokenizer = self.processor.tokenizer
        
        # Model configuration
        self.model.to(self.device)
        self.model.eval()
        self.processor.tokenizer.padding_side = "left"

    def _prepare_images(self, batch: List[List[Image.Image]], normalize=True) -> torch.Tensor:
        """修正后的图像预处理方法"""
        processor = self.processor.image_processor
        # 使用processor直接处理PIL图像
        processed_images = [
            processor(example[0], do_normalize=normalize, return_tensors="pt")["pixel_values"]
            for example in batch
        ]
        return torch.cat(processed_images, dim=0).to(self.device)

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        pixel_values = self._prepare_images(batch_images)
        
        inputs = self.processor(
            text=batch_text,
            images=None,
            padding=True,
            truncation=True,
            max_length=2000,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty
            )

        return self.processor.batch_decode(outputs, skip_special_tokens=True)

    def get_outputs_attack(
        self,
        attack: torch.Tensor,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        # 统一处理流程
        pixel_values = self._prepare_images(batch_images, normalize=False)
        pixel_values = transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)(
            pixel_values + attack.to(self.device)
        )
        
        inputs = self.processor(
            text=batch_text,
            images=None,
            padding=True,
            truncation=True,
            max_length=2000,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty
            )

        return self.processor.batch_decode(outputs, skip_special_tokens=True)

    def get_vqa_prompt(self, question: str, answer: str = None) -> tuple:
        """返回问题和答案的元组"""
        question_part = f"Question:{question}"
        answer_part = f"Answer:{answer if answer is not None else ''}"
        return question_part, answer_part

    def get_caption_prompt(self, caption: str = None) -> str:
        return ""  # BLIP-2 doesn't need special caption prompt

    def get_classification_prompt(self, class_str: str = None) -> str:
        return ""  # BLIP-2 doesn't need special classification prompt
    
    def set_custom_embeddings(self, embeddings):
        """临时替换embedding层"""
        if not hasattr(self, '_original_embedding'):
            self._original_embedding = self.model.get_input_embeddings()
        
        class HookedEmbedding(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.original = original
                self.custom = None
            
            def forward(self, input_ids):
                return self.custom if self.custom is not None else self.original(input_ids)
        
        if not hasattr(self.model, '_hooked_embedding'):
            self.model._hooked_embedding = HookedEmbedding(self._original_embedding)
            self.model.set_input_embeddings(self.model._hooked_embedding)
        
        self.model._hooked_embedding.custom = embeddings

    def clear_custom_embeddings(self):
        """恢复原始embedding层"""
        if hasattr(self.model, '_hooked_embedding'):
            self.model._hooked_embedding.custom = None