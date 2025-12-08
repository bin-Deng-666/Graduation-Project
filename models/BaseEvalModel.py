from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import torch

class BaseEvalModel(ABC):
    """Abstract base class for model evaluation interfaces."""
    
    @abstractmethod
    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        """Generate outputs from model."""
        pass
    
    @abstractmethod
    def get_vqa_prompt(self, question: str, answer: str = None) -> str:
        """Generate VQA prompt."""
        pass
    
    @abstractmethod
    def get_caption_prompt(self, caption: str = None) -> str:
        """Generate caption prompt."""
        pass
    
    @abstractmethod
    def get_classification_prompt(self, class_str: str = None) -> str:
        """Generate classification prompt."""
        pass
    
    # 可能还包含其他攻击相关的方法
    @abstractmethod
    def get_outputs_attack(
        self,
        attack: torch.Tensor,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        """Generate outputs with attack."""
        pass