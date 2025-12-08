import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.my_eval_datasets import COCO_Dataset

def test_coco_dataset():
    # 配置数据路径 - 指向项目根目录下的data文件夹
    parser = argparse.ArgumentParser()
    # 修改数据路径，指向项目根目录下的data文件夹
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    # COCO数据集参数配置
    parser.add_argument("--coco_train_image_dir", 
                      type=str, 
                      default=os.path.join(data_root, "train2014"),
                      help="COCO训练集图片目录路径")
    
    parser.add_argument("--coco_train_questions",
                      type=str,
                      default=os.path.join(data_root, "v2_OpenEnded_mscoco_train2014_questions.json"),
                      help="COCO训练集问题文件路径")
    
    parser.add_argument("--coco_train_annotations",
                      type=str,
                      default=os.path.join(data_root, "v2_mscoco_train2014_annotations.json"),
                      help="COCO训练集标注文件路径")
    
    parser.add_argument("--coco_val_image_dir",
                      type=str,
                      default=os.path.join(data_root, "val2014"),
                      help="COCO验证集图片目录路径")
    
    parser.add_argument("--coco_val_questions",
                      type=str,
                      default=os.path.join(data_root, "filtered_v2_OpenEnded_mscoco_val2014_questions.json"),
                      help="COCO验证集问题文件路径")
    
    parser.add_argument("--coco_val_annotations",
                      type=str,
                      default=os.path.join(data_root, "filtered_v2_mscoco_val2014_annotations.json"),
                      help="COCO验证集标注文件路径")
    
    args = parser.parse_args([])  # 使用默认值
    
    # 添加路径检查
    print("路径存在性检查:")
    paths_to_check = {
        "训练图片目录": args.coco_train_image_dir,
        "训练问题文件": args.coco_train_questions,
        "训练标注文件": args.coco_train_annotations,
        "验证图片目录": args.coco_val_image_dir,
        "验证问题文件": args.coco_val_questions,
        "验证标注文件": args.coco_val_annotations
    }
    
    all_paths_exist = True
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        print(f"{name}: {path} - {'存在' if exists else '不存在'}")
        if not exists:
            all_paths_exist = False
    
    if not all_paths_exist:
        print("\n警告: 部分数据文件不存在，请先下载或移动到正确位置。")
    
    # 测试数据集加载
    try:
        # 测试训练集
        train_dataset = COCO_Dataset(
            image_dir_path=args.coco_train_image_dir,
            question_path=args.coco_train_questions,
            annotations_path=args.coco_train_annotations,
            is_train=True
        )
        
        # 测试验证集
        val_dataset = COCO_Dataset(
            image_dir_path=args.coco_val_image_dir,
            question_path=args.coco_val_questions,
            annotations_path=args.coco_val_annotations,
            is_train=False
        )
        
        # 基本信息测试
        print(f"\n训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        # 测试样本获取
        sample = train_dataset[0]
        print("\n训练集第一个样本:")
        print(f"图片类型: {type(sample['image'])}")  # 显示图片类型而不是图片本身
        print(f"问题: {sample['question']}")
        print(f"答案: {sample['answers']}")
        print(f"问题ID: {sample['question_id']}")
        
    except FileNotFoundError as e:
        print(f"\n文件未找到错误: {str(e)}")
        print("请确保所有COCO数据集文件都已下载并放在正确位置")
    except json.JSONDecodeError as e:
        print(f"\nJSON解析错误: {str(e)}")
        print("请检查JSON文件格式是否正确")
    except Exception as e:
        print(f"\nCOCO数据集加载失败: {str(e)}")
        print("请检查:")
        print("1. 数据文件路径是否正确")
        print("2. 数据文件是否存在")
        print("3. 数据文件格式是否符合要求")

if __name__ == "__main__":
    test_coco_dataset()