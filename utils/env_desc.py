import json  
import base64
from openai import OpenAI
from io import BytesIO
from PIL import Image
from utils.my_attack_tool import load_dataset, get_subset
import time
import os

# 初始化结果字典和输出路径
results = {}
output_path = "data/id_to_environment.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 加载数据集并获取子集
_, test_dataset = load_dataset()
test_dataset = get_subset(dataset=test_dataset, frac=1)

client = OpenAI(
    api_key="sk-drszqbypcxdnqjurtdqlwlajlzlizpgcshwudnqwnjkayiui",
    base_url="https://api.siliconflow.cn/v1"
)

def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

for item in test_dataset:
    try:
        image = item["image"]
        base64_image = image_to_base64(image)
        
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-VL-72B-Instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": "Describe the environment in the image using one adjective followed by one noun."}
                        ]
                    }],
                    stream=False
                )
                
                # 存储结果到字典
                raw_response = response.choices[0].message.content
                processed_response = raw_response.lower().replace('.', '').strip()
                results[str(item["image_id"])] = processed_response
                print(f"Processed Image ID: {item['image_id']}, Response: {processed_response}")
                break
                
            except Exception as e:
                if attempt == 2:
                    print(f"Failed Image ID: {item['image_id']}, Error: {str(e)}")
                time.sleep(1)
                
    except Exception as e:
        print(f"Skipped Image ID: {item['image_id']}, Error: {str(e)}")
        continue

# 保存结果到JSON文件
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {output_path}")