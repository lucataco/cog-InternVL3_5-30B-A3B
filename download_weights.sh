#!/bin/bash
# Script to download InternVL3.5-30B-A3B model weights

set -e

echo "======================================"
echo "InternVL3.5-30B-A3B Weight Downloader"
echo "======================================"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

# Set the checkpoint directory
CHECKPOINT_DIR="./checkpoints"

# Check if checkpoints already exist
if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
    echo "Warning: Checkpoints directory already exists and is not empty."
    read -p "Do you want to continue and potentially overwrite existing files? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Download cancelled."
        exit 0
    fi
fi

# Create checkpoints directory
mkdir -p "$CHECKPOINT_DIR"

echo "Downloading model weights..."
echo "This may take a while (~60GB download)..."
echo ""

# Download the model
huggingface-cli download \
    OpenGVLab/InternVL3_5-30B-A3B-HF \
    --local-dir "$CHECKPOINT_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "======================================"
echo "Download complete!"
echo "======================================"
echo ""
echo "Model weights are saved in: $CHECKPOINT_DIR"
echo "You can now run: cog build"
echo ""

