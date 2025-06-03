#!/bin/bash

# Zero-shot Classification Evaluation Script
# 
#   ./run_zero_shot_eval.sh              
#   ./run_zero_shot_eval.sh 0            

echo "Starting Zero-shot Classification Evaluation..."

if [ $# -eq 1 ]; then
    gpu_choice=$1
    if [ "$gpu_choice" = "auto" ]; then
        echo "Using all available GPUs (auto mode)"
    else
        echo "Using GPU: $gpu_choice (from command line)"
        export CUDA_VISIBLE_DEVICES=$gpu_choice
    fi
else
    echo "=== GPU Selection ==="
    if command -v nvidia-smi &> /dev/null; then
        echo "Available GPUs:"
        nvidia-smi --list-gpus
        echo ""
        echo "GPU Memory Usage:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
        echo ""
    else
        echo "nvidia-smi not found, assuming CUDA is available"
    fi

    echo "Please select GPU to use:"
    echo "  - Enter GPU number (0, 1, 2, etc.)"
    echo "  - Press Enter for default (GPU 0)"
    echo "  - Enter 'auto' to use all available GPUs"
    read -p "Your choice: " gpu_choice

    if [ -z "$gpu_choice" ]; then
        gpu_choice=0
        echo "Using default GPU: 0"
        export CUDA_VISIBLE_DEVICES=0
    elif [ "$gpu_choice" = "auto" ]; then
        echo "Using all available GPUs"
    else
        echo "Using GPU: $gpu_choice"
        export CUDA_VISIBLE_DEVICES=$gpu_choice
    fi
    echo "========================"
fi

SCRIPT_PATH="../val/zero_shot_evaluation.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    echo "Please make sure the Python file is in the current directory"
    exit 1
fi


python3 "$SCRIPT_PATH" \
    --model_name "ViT-B-16" \
    --pretrained_path "/path/to/PathoCLIP-B.pt" \
    --batch_size 64 \
    --num_workers 4 \
    --output_dir "./results" \
    --log_dir "./logs" \
    --detailed_metrics

echo "Evaluation completed!"
echo ""
echo "=== Results Location ==="
echo "Results saved in: ./results/"
echo "Logs saved in: ./logs/"
echo ""
echo "To view the latest results:"
echo "  ls -la ./results/"
echo "  cat ./results/zero_shot_summary_*.json"