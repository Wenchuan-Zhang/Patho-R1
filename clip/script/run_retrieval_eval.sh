#!/bin/bash

# CLIP Model Image-to-Text Retrieval Evaluation
#   ./run_clip_retrieval.sh              
#   ./run_clip_retrieval.sh 0                   

echo "Starting CLIP Text-to-Image Retrieval Evaluation..."

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

SCRIPT_PATH="../val/retrieval_evaluation.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    echo "Please make sure the Python file is in the current directory"
    exit 1
fi

MODEL_NAME="ViT-B-16"
PRETRAINED_PATH="/path/to/PathoCLIP-B.pt"
CAPTIONS_JSON="/path/to/captions.json"
IMAGES_DIR="/path/to/images"

echo "=== Checking Required Files ==="
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Warning: Pretrained model not found at: $PRETRAINED_PATH"
    echo "Please modify PRETRAINED_PATH in this script"
fi

if [ ! -f "$CAPTIONS_JSON" ]; then
    echo "Warning: Captions JSON not found at: $CAPTIONS_JSON"
    echo "Please modify CAPTIONS_JSON in this script"
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Warning: Images directory not found at: $IMAGES_DIR"
    echo "Please modify IMAGES_DIR in this script"
fi
echo "========================"

echo "Starting evaluation with the following parameters:"
echo "  Model: $MODEL_NAME"
echo "  Pretrained Path: $PRETRAINED_PATH"
echo "  Captions JSON: $CAPTIONS_JSON"
echo "  Images Directory: $IMAGES_DIR"
echo ""

python3 "$SCRIPT_PATH" \
    --model-name "$MODEL_NAME" \
    --pretrained-path "$PRETRAINED_PATH" \
    --captions-json "$CAPTIONS_JSON" \
    --images-dir "$IMAGES_DIR" \
    --batch-size 64 \
    --text-batch-size 256 \
    --device auto \
    --num-workers 4 \
    --recall-k 1 5 10 20 \
    --output-dir "./results" \
    --save-features \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo ""
    echo "=== Results Location ==="
    echo "Results saved in: ./results/"
    echo "Features saved in: ./results/features/"
    echo ""
    echo "To view the latest results:"
    echo "  ls -la ./results/"
    echo "  cat ./results/clip_evaluation_*.json | jq '.recall_metrics'"
else
    echo ""
    echo "Evaluation failed. Check the error messages above."
    echo "Common issues:"
    echo "  1. Check if all required paths exist and are accessible"
    echo "  2. Ensure you have enough GPU memory"
    echo "  3. Verify the captions JSON format matches expected structure"
    echo "  4. Check if all required Python packages are installed"
fi