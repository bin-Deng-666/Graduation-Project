#!/bin/bash

# 设置错误时退出
set -e

# 定义默认参数
initialize_default_parameters() {
    # 模型和方法参数
    MODEL_NAME="blip2"
    METHOD="baseline"
    
    # 评估参数
    TARGET_TEXT="Unknown"
    EVAL_BATCH_SIZE=4
    MAX_GEN_LENGTH=50
    NUM_BEAMS=3
    LENGTH_PENALTY=-1.0
    
    # 硬件配置
    DEVICE=0
}

# 解析命令行参数
parse_command_line_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model_name|-m)
                MODEL_NAME="$2"
                shift 2
                ;;
            --method)
                METHOD="$2"
                shift 2
                ;;
            --device|-d)
                DEVICE="$2"
                shift 2
                ;;
            --target_text|-t)
                TARGET_TEXT="$2"
                shift 2
                ;;
            --eval_batch_size|-b)
                EVAL_BATCH_SIZE="$2"
                shift 2
                ;;
            --max_generation_length|-l)
                MAX_GEN_LENGTH="$2"
                shift 2
                ;;
            --num_beams|-n)
                NUM_BEAMS="$2"
                shift 2
                ;;
            --length_penalty|-p)
                LENGTH_PENALTY="$2"
                shift 2
                ;;
            *)
                echo "未知参数: $1"
                exit 1
                ;;
        esac
    done
}

# 显示配置信息
display_configuration() {
    echo "=== 评估配置信息 ==="
    echo "模型名称: $MODEL_NAME"
    echo "攻击方法: $METHOD"
    echo "设备: $DEVICE"
    echo "目标文本: $TARGET_TEXT"
    echo "评估批大小: $EVAL_BATCH_SIZE"
    echo "最大生成长度: $MAX_GEN_LENGTH"
    echo "Beam数量: $NUM_BEAMS"
    echo "长度惩罚: $LENGTH_PENALTY"
    echo "===================="
}

# 执行评估任务
execute_evaluation() {
    echo "开始执行评估任务..."
    python evaluate_attack.py \
        --model_name "$MODEL_NAME" \
        --method "$METHOD" \
        --device "$DEVICE" \
        --target_text "$TARGET_TEXT" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --max_generation_length "$MAX_GEN_LENGTH" \
        --num_beams "$NUM_BEAMS" \
        --length_penalty "$LENGTH_PENALTY"
    
    echo "评估任务已完成"
}

# 主函数
main() {
    # 初始化默认参数
    initialize_default_parameters
    
    # 解析命令行参数
    parse_command_line_arguments "$@"
    
    # 显示配置信息
    display_configuration
    
    # 执行评估任务
    execute_evaluation
}

# 调用主函数
main "$@"