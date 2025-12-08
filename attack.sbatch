#!/bin/bash

# SLURM批处理作业配置
#SBATCH --job-name=adv_attack
#SBATCH --partition=cluster-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/attack_%j.log

# 设置错误时退出
set -e

# 定义默认参数
initialize_default_parameters() {
    # 模型和方法参数
    MODEL_NAME="blip2"
    METHOD="embed_noise"
    
    # 对抗攻击参数
    PROMPT_NUM=25
    ADV_LENGTH=10
    ITERS=500
    FRACTION=0.05
    
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
            --prompt_num|-p)
                PROMPT_NUM="$2"
                shift 2
                ;;
            --adversarial_length|-l)
                ADV_LENGTH="$2"
                shift 2
                ;;
            --device|-d)
                DEVICE="$2"
                shift 2
                ;;
            --iters|-i)
                ITERS="$2"
                shift 2
                ;;
            --fraction|-f)
                FRACTION="$2"
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
    echo "=== 对抗攻击配置信息 ==="
    echo "任务ID: $SLURM_JOB_ID"
    echo "模型名称: $MODEL_NAME"
    echo "攻击方法: $METHOD"
    echo "提示词数量: $PROMPT_NUM"
    echo "对抗长度: $ADV_LENGTH"
    echo "设备: $DEVICE"
    echo "迭代次数: $ITERS"
    echo "分数: $FRACTION"
    echo "========================"
}

# 执行对抗攻击任务
execute_attack() {
    echo "开始执行对抗攻击任务..."
    python train.py \
        --model_name "$MODEL_NAME" \
        --method "$METHOD" \
        --prompt_num "$PROMPT_NUM" \
        --adversarial_length "$ADV_LENGTH" \
        --device "$DEVICE" \
        --iters "$ITERS" \
        --fraction "$FRACTION"
    
    echo "对抗攻击任务已完成"
}

# 主函数
main() {
    # 确保日志目录存在
    mkdir -p logs
    
    # 初始化默认参数
    initialize_default_parameters
    
    # 解析命令行参数
    parse_command_line_arguments "$@"
    
    # 显示配置信息
    display_configuration
    
    # 执行攻击任务
    execute_attack
}

# 调用主函数
main "$@"
