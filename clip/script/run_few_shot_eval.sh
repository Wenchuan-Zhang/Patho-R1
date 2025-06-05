#!/bin/bash

# Few-shot Learning Evaluation Script
#   ./run_few_shot_eval.sh              
#   ./run_few_shot_eval.sh 0            


echo "Starting Few-shot Learning Evaluation..."

if [ $# -eq 1 ]; then
    gpu_choice=$1
    if [ "$gpu_choice" = "-1" ]; then
        echo "Using CPU (gpu_id=-1)"
        GPU_ID=-1
    else
        echo "Using GPU: $gpu_choice (from command line)"
        GPU_ID=$gpu_choice
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
    echo "  - Enter -1 to use CPU"
    echo "  - Press Enter for default (GPU 0)"
    read -p "Your choice: " gpu_choice

    # Processing input
    if [ -z "$gpu_choice" ]; then
        GPU_ID=0
        echo "Using default GPU: 0"
        export CUDA_VISIBLE_DEVICES=0
    elif [ "$gpu_choice" = "-1" ]; then
        GPU_ID=-1
        echo "Using CPU"
    else
        GPU_ID=$gpu_choice
        echo "Using GPU: $gpu_choice"
        export CUDA_VISIBLE_DEVICES=$gpu_choice
    fi
    echo "========================"
fi

# Set the Python script path
SCRIPT_PATH="../val/few_shot_evaluation.py"

# Check if the Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    echo "Please make sure the Python file is in the current directory"
    exit 1
fi

DATASET_NAME="LC-lung"
TRAIN_ROOT="/path/to/your/train/data"
TEST_ROOT="/path/to/your/test/data"
MODEL_NAME="ViT-B-16"
PRETRAINED_PATH="/path/to/your/pretrained/model.pt"

echo "=== Checking Required Files and Directories ==="
if [ ! -d "$TRAIN_ROOT" ]; then
    echo "Warning: Training data directory not found at: $TRAIN_ROOT"
    echo "Please modify TRAIN_ROOT in this script"
fi

if [ ! -d "$TEST_ROOT" ]; then
    echo "Warning: Test data directory not found at: $TEST_ROOT"
    echo "Please modify TEST_ROOT in this script"
fi

if [ "$PRETRAINED_PATH" != "openai" ] && [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Warning: Pretrained model not found at: $PRETRAINED_PATH"
    echo "Will use default OpenAI pretrained weights"
    PRETRAINED_PATH="openai"
fi
echo "========================"

# Set Shot Count List
SHOT_LIST="2 8 16 32 64 128"

echo "Starting evaluation with the following parameters:"
echo "  Dataset: $DATASET_NAME"
echo "  Model: $MODEL_NAME"
echo "  Pretrained Path: $PRETRAINED_PATH"
echo "  Train Root: $TRAIN_ROOT"
echo "  Test Root: $TEST_ROOT"
echo "  GPU ID: $GPU_ID"
echo "  Shot List: $SHOT_LIST"
echo ""

python3 "$SCRIPT_PATH" \
    --train_root "$TRAIN_ROOT" \
    --test_root "$TEST_ROOT" \
    --dataset "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --pretrained_path "$PRETRAINED_PATH" \
    --gpu_id $GPU_ID \
    --shot_list $SHOT_LIST \
    --n_subsets 10 \
    --global_seed 42 \
    --batch_size 64 \
    --num_workers 4 \
    --reg_strength 1.0 \
    --max_iter 1000 \
    --output_prefix "few_shot_evaluation"

# Check the execution results
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo ""
    echo "=== Results Location ==="
    echo "Results saved as:"
    echo "  Excel: few_shot_evaluation_${DATASET_NAME}_summary.xlsx"
    echo "  CSV: few_shot_evaluation_${DATASET_NAME}_summary.csv"
    echo "  Log: few_shot_evaluation_${DATASET_NAME}.log"
    echo ""
    echo "To view the results:"
    echo "  # View Excel file (if you have pandas installed):"
    echo "  python3 -c \"import pandas as pd; print(pd.read_excel('few_shot_evaluation_${DATASET_NAME}_summary.xlsx'))\""
    echo ""
    echo "  # View CSV file:"
    echo "  cat few_shot_evaluation_${DATASET_NAME}_summary.csv"
    echo ""
    echo "  # View log file:"
    echo "  tail -n 50 few_shot_evaluation_${DATASET_NAME}.log"
    echo ""
    echo "=== Quick Results Summary ==="
    if [ -f "few_shot_evaluation_${DATASET_NAME}_summary.csv" ]; then
        echo "Mean accuracy by shot:"
        echo "Shot | Mean Acc | Std Acc"
        echo "----|----------|--------"
        tail -n +2 "few_shot_evaluation_${DATASET_NAME}_summary.csv" | cut -d',' -f2,12,13 | while IFS=',' read shot mean_acc std_acc; do
            printf "%4s | %8.2f | %7.2f\n" "$shot" "$mean_acc" "$std_acc"
        done
    fi
else
    echo ""
    echo "Evaluation failed. Check the error messages above."
    echo ""
    echo "Common issues and solutions:"
    echo "  1. Path Issues:"
    echo "     - Check if train_root and test_root paths exist"
    echo "     - Ensure the directories contain proper ImageFolder structure"
    echo "     - Verify pretrained model path (or use 'openai' for default)"
    echo ""
    echo "  2. Memory Issues:"
    echo "     - Try reducing batch_size (current: 64)"
    echo "     - Use CPU instead of GPU: ./run_few_shot_eval.sh -1"
    echo "     - Reduce the number of shots in SHOT_LIST"
    echo ""
    echo "  3. Package Issues:"
    echo "     - Install required packages: pip install open_clip_torch pandas openpyxl"
    echo "     - Check if sklearn and torch are properly installed"
    echo ""
    echo "  4. Data Issues:"
    echo "     - Ensure your data follows ImageFolder structure:"
    echo "       train_root/class1/image1.jpg, train_root/class2/image2.jpg, etc."
    echo "     - Check if all classes have sufficient samples for few-shot learning"
    echo ""
    echo "Check the log file for detailed error information:"
    echo "  tail -n 100 few_shot_evaluation_${DATASET_NAME}.log"
fi