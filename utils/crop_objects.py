from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import traceback
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入数据集加载模块
from utils.attack_tool import load_dataset, get_subset


# 配置参数
class Config:
    # 最小物体尺寸（宽度和高度），确保足够的内容用于embedding
    MIN_OBJECT_WIDTH = 50
    MIN_OBJECT_HEIGHT = 50
    # 最小置信度阈值
    MIN_CONFIDENCE = 0.5
    # YOLO模型路径
    YOLO_MODEL_PATH = "models/YOLO/yolov8m.pt"
    # 输出目录
    OUTPUT_DIR = "data/cropped_objects"


def is_valid_object(box, min_width=Config.MIN_OBJECT_WIDTH, min_height=Config.MIN_OBJECT_HEIGHT):
    """
    检查裁剪区域是否满足最小尺寸要求
    
    Args:
        box: 边界框坐标 [x1, y1, x2, y2]
        min_width: 最小宽度要求
        min_height: 最小高度要求
    
    Returns:
        bool: 如果满足尺寸要求返回True，否则返回False
    """
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    return width >= min_width and height >= min_height


def get_main_objects(detections, top_k=8):
    """
    根据置信度和面积选择主要物体
    
    Args:
        detections: 检测结果列表，每个元素为 (box, class_id, conf, area)
        top_k: 返回前k个主要物体
    
    Returns:
        list: 筛选后的主要物体列表
    """
    # 首先过滤掉不满足尺寸要求的物体
    valid_detections = [(box, class_id, conf, area) for box, class_id, conf, area in detections 
                        if is_valid_object(box)]
    
    if not valid_detections:
        return []
    
    # 先按置信度排序，再按面积排序
    valid_detections.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    # 返回前k个主要物体
    return valid_detections[:top_k]


def process_image(model, image_data, output_dir):
    """
    处理单个图像，检测并裁剪主要物体
    
    Args:
        model: YOLO模型实例
        image_data: 图像数据字典，包含'image'和'image_id'
        output_dir: 输出目录
    
    Returns:
        int: 保存的物体数量
    """
    try:
        image = image_data["image"]
        img_name = str(image_data["image_id"])  # 使用image_id作为文件夹名
        
        # 为每张图片创建单独文件夹
        img_output_dir = os.path.join(output_dir, img_name)
        # 先删除已存在的文件夹，确保干净的环境
        if os.path.exists(img_output_dir):
            import shutil
            shutil.rmtree(img_output_dir)
            print(f"  Deleted existing folder: {img_output_dir}")
        # 重新创建文件夹
        os.makedirs(img_output_dir, exist_ok=True)
        
        # 将PIL图像转换为OpenCV格式
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 运行YOLO推理，使用内置的NMS过滤重复检测
        results = model(img, conf=Config.MIN_CONFIDENCE)  # 直接传入置信度阈值
        
        # 收集所有检测结果
        all_detections = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, class_id, conf in zip(boxes, class_ids, confidences):
                # 计算面积
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                all_detections.append((box, class_id, conf, area))
        
        # 获取主要物体，使用NMS过滤重叠区域
        main_objects = get_main_objects(all_detections)
        
        if not main_objects:
            print(f"No valid objects found for Image: {img_name}")
            return 0
        
        # 保存主要物体
        saved_objects = 0
        saved_areas = set()  # 用于记录已保存区域，避免重复
        
        for i, (box, class_id, conf, area) in enumerate(main_objects):
            x1, y1, x2, y2 = map(int, box)
            
            # 检查是否与已保存区域有明显重叠（用边界框坐标作为标识符）
            # 为了避免浮点数精度问题，使用坐标的近似值
            area_key = (round(x1/10)*10, round(y1/10)*10, round(x2/10)*10, round(y2/10)*10)
            
            if area_key not in saved_areas:
                crop_img = img[y1:y2, x1:x2]
                
                # 获取类别名称
                class_name = model.names[int(class_id)]
                width = x2 - x1
                height = y2 - y1
                
                # 保存为: 图片ID文件夹/类别名称_尺寸_置信度_索引.jpg
                output_path = os.path.join(img_output_dir, 
                                          f"{class_name}_{width}x{height}_{conf:.2f}_{i}.jpg")
                
                # 处理重名文件
                counter = 1
                while os.path.exists(output_path):
                    output_path = os.path.join(img_output_dir, 
                                              f"{class_name}_{width}x{height}_{conf:.2f}_{i}_{counter}.jpg")
                    counter += 1
                
                cv2.imwrite(output_path, crop_img)
                saved_areas.add(area_key)
                saved_objects += 1
                print(f"  Saved main object: {class_name} ({width}x{height}, conf: {conf:.2f})")
            else:
                print(f"  Skipped duplicate object at position: {x1, y1, x2, y2}")
        
        print(f"Processed Image: {img_name}, saved {saved_objects} main objects")
        return saved_objects
        
    except Exception as e:
        img_name = str(image_data.get("image_id", "unknown"))
        print(f"Error processing Image: {img_name}, Error: {str(e)}")
        traceback.print_exc()  # 打印详细错误堆栈
        return 0


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='裁剪数据集中检测到的物体')
    parser.add_argument('--model', type=str, default=Config.YOLO_MODEL_PATH, help='YOLO模型路径')
    parser.add_argument('--output', type=str, default=Config.OUTPUT_DIR, help='裁剪物体的输出目录')
    parser.add_argument('--conf-threshold', type=float, default=Config.MIN_CONFIDENCE, 
                        help='物体置信度阈值，低于此值的物体将被过滤')
    parser.add_argument('--subset-frac', type=float, default=1.0, 
                        help='数据集子集比例 (0.0-1.0)')
    parser.add_argument('--iou-threshold', type=float, default=0.7, 
                        help='重叠区域IoU阈值，高于此值的区域将被视为重复 (0.0-1.0)')
    args = parser.parse_args()
    
    # 更新配置
    Config.MIN_CONFIDENCE = args.conf_threshold
    # 存储IoU阈值用于get_main_objects函数
    args.iou_threshold = args.iou_threshold
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 打印配置信息
    print("配置信息:")
    print(f"  模型路径: {args.model}")
    print(f"  输出目录: {args.output}")
    print(f"  最小物体宽度: {Config.MIN_OBJECT_WIDTH}px")
    print(f"  最小物体高度: {Config.MIN_OBJECT_HEIGHT}px")
    print(f"  置信度阈值: {Config.MIN_CONFIDENCE}")
    print(f"  数据集子集比例: {args.subset_frac}")
    print(f"  重叠区域IoU阈值: {args.iou_threshold}")
    
    # 加载模型
    print("\n加载YOLO模型...")
    model = YOLO(args.model)
    print("模型加载完成！")
    
    # 加载数据集并获取子集
    print("\n加载数据集...")
    _, test_dataset = load_dataset()
    test_dataset = get_subset(dataset=test_dataset, frac=args.subset_frac)
    print(f"数据集加载完成！子集包含 {len(test_dataset)} 张图像")
    
    # 处理数据集
    print("\n开始处理数据集...")
    total_processed = 0
    total_saved = 0
    
    for item in test_dataset:
        saved_count = process_image(model, item, args.output)
        total_processed += 1
        total_saved += saved_count
    
    print(f"\n处理完成！")
    print(f"总共处理了 {total_processed} 张图像")
    print(f"总共保存了 {total_saved} 个有效物体")


if __name__ == "__main__":
    main()