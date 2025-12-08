import json
import os

from PIL import Image
from torch.utils.data import Dataset

class COCO_Dataset(Dataset):
    def __init__(self, image_dir_path, question_path, annotations_path, is_train):
        self.image_dir_path = image_dir_path
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.is_train = is_train

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]

        img_path = os.path.join(
            self.image_dir_path,
            f"COCO_train2014_{question['image_id']:012d}.jpg" if self.is_train 
            else f"COCO_val2014_{question['image_id']:012d}.jpg"
        )
        
        image = Image.open(img_path).convert("RGB")
        return {
            "image": image,
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
            "image_id": str(question["image_id"])
        }

