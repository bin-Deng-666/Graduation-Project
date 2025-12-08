import torch
from models.myblip2 import EvalModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def test_custom_embeddings(eval_model, device="cuda:0"):
    """测试自定义嵌入功能"""
    print("\n=== 开始测试自定义嵌入功能 ===")
    
    # 准备测试数据
    test_text = "测试文本"
    test_inputs = eval_model.processor(
        text=[test_text],
        padding=True,
        truncation=True,
        max_length=2000,
        return_tensors="pt"
    ).to(device)
    
    # 准备随机图像输入
    test_image = torch.rand(1, 3, 224, 224).to(device)
    
    # 1. 原始模型计算loss
    original_outputs = eval_model.model(
        input_ids=test_inputs.input_ids,
        pixel_values=test_image,
        attention_mask=test_inputs.attention_mask,
        labels=test_inputs.input_ids
    )
    print(f"原始模型loss: {original_outputs.loss.item()}")
    
    # 2. 使用set_custom_embeddings计算loss
    text_emb = eval_model.model.get_input_embeddings()(test_inputs.input_ids)
    eval_model.set_custom_embeddings(text_emb)
    custom_outputs = eval_model.model(
        input_ids=test_inputs.input_ids,
        pixel_values=test_image,
        attention_mask=test_inputs.attention_mask,
        labels=test_inputs.input_ids
    )
    eval_model.clear_custom_embeddings()
    print(f"自定义嵌入loss(应与原始相同): {custom_outputs.loss.item()}")
    
    # 3. 使用不同的自定义嵌入计算loss
    random_emb = torch.rand_like(text_emb).to(device)
    eval_model.set_custom_embeddings(random_emb)
    random_outputs = eval_model.model(
        input_ids=test_inputs.input_ids,
        pixel_values=test_image,
        attention_mask=test_inputs.attention_mask,
        labels=test_inputs.input_ids
    )
    eval_model.clear_custom_embeddings()
    print(f"随机嵌入loss(应与前两者不同): {random_outputs.loss.item()}")

    # 4. 新增：使用随机input_ids测试
    vocab_size = eval_model.processor.tokenizer.vocab_size
    random_ids = torch.randint(
        low=100,  # 跳过特殊token
        high=vocab_size-1,
        size=test_inputs.input_ids.shape,
        device=device
    )
    
    # 使用原始嵌入但随机input_ids
    eval_model.set_custom_embeddings(text_emb)
    random_id_outputs = eval_model.model(
        input_ids=random_ids,  # 使用随机input_ids
        pixel_values=test_image,
        attention_mask=test_inputs.attention_mask,
        labels=test_inputs.input_ids  # 保持原始labels
    )
    eval_model.clear_custom_embeddings()
    print(f"随机input_ids + 原始嵌入的loss: {random_id_outputs.loss.item()}")


    # 5. 随机替换50%的token为pad_token
    pad_token_id = eval_model.processor.tokenizer.pad_token_id or 0  # 默认为0
    print(f"\n使用的pad_token_id: {pad_token_id}")

    original_ids = test_inputs.input_ids.clone()
    modified_ids = original_ids.clone()
    
    # 随机选择50%的位置替换为pad_token
    replace_mask = torch.rand(original_ids.size(1), device=device) < 0.1
    modified_ids[0, replace_mask] = pad_token_id
    
    # 计算loss
    pad_outputs = eval_model.model(
        input_ids=modified_ids,
        pixel_values=test_image,
        attention_mask=test_inputs.attention_mask,
        labels=test_inputs.input_ids  # 保持原始labels
    )
    
    print(f"\n[Pad Token测试]")
    print(f"原始input_ids: {original_ids[0].tolist()}")
    print(f"修改后input_ids: {modified_ids[0].tolist()}")
    print(f"原始loss: {original_outputs.loss.item():.4f}")
    print(f"替换后loss: {pad_outputs.loss.item():.4f}")
    
    print("=== 测试结束 ===\n")

if __name__ == "__main__":
    # 初始化模型
    model_args = {
        "processor_path": "Salesforce/blip2-opt-2.7b",
        "lm_path": "Salesforce/blip2-opt-2.7b", 
        "device": 0
    }
    eval_model = EvalModel(model_args)
    
    # 运行测试
    test_custom_embeddings(eval_model, device=f"cuda:{model_args['device']}")