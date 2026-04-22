#!/bin/bash
# Build ComfyUI Blackwell Kernels on Linux
# Requirements: CUDA 13.2 Toolkit, Python 3.10+, PyTorch 2.5+
set -e

echo "=== ComfyUI Blackwell Kernels — Build from Source ==="

# Auto-detect CUDA 13.2
if [ -z "$CUDA_HOME" ]; then
    for p in /usr/local/cuda-13.2 /usr/local/cuda-13 /usr/local/cuda; do
        if [ -x "$p/bin/nvcc" ]; then
            export CUDA_HOME="$p"
            break
        fi
    done
fi

if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: CUDA Toolkit not found. Set CUDA_HOME or install CUDA 13.2"
    exit 1
fi

echo "CUDA_HOME=$CUDA_HOME"
$CUDA_HOME/bin/nvcc --version | head -1

export TORCH_CUDA_ARCH_LIST="12.0a"
pip install -e .

echo ""
echo "=== BUILD SUCCESSFUL ==="
echo "Copy this folder into ComfyUI/custom_nodes/ to activate."
