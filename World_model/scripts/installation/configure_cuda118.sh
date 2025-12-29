#!/bin/bash

# CUDA 11.8 Configuration Script for World4Omni
# This script configures CUDA_HOME and related environment variables for CUDA 11.8

set -e

echo "🔧 CUDA 11.8 Configuration Script"
echo "================================="

# Check if CUDA 11.8 is installed
CUDA_PATH="/usr/local/cuda-11.8"
if [ ! -d "$CUDA_PATH" ]; then
    echo "❌ CUDA 11.8 not found at $CUDA_PATH"
    echo "Please install CUDA 11.8 first:"
    echo "  - Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive"
    echo "  - Or use: sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

echo "✅ CUDA 11.8 found at: $CUDA_PATH"

# Set environment variables for current session
export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "🔧 Environment variables set for current session:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  PATH: $CUDA_HOME/bin added"
echo "  LD_LIBRARY_PATH: $CUDA_HOME/lib64 added"

# Check if nvcc is accessible
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "✅ nvcc accessible, CUDA version: $CUDA_VERSION"
else
    echo "⚠️  Warning: nvcc not found in PATH"
fi

# Add to shell configuration files
SHELL_CONFIGS=("~/.bashrc" "~/.bash_profile" "~/.profile")

for config in "${SHELL_CONFIGS[@]}"; do
    config_file="${config/#\~/$HOME}"
    if [ -f "$config_file" ]; then
        if ! grep -q "CUDA_HOME=/usr/local/cuda-11.8" "$config_file" 2>/dev/null; then
            echo "📝 Adding CUDA 11.8 configuration to $config_file..."
            cat >> "$config_file" << 'EOF'

# CUDA 11.8 Configuration for World4Omni
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
            echo "✅ Configuration added to $config_file"
        else
            echo "✅ Configuration already exists in $config_file"
        fi
    fi
done

# Activate grounding-sam environment for testing
echo ""
echo "🔧 Activating grounding-sam environment for testing..."
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate grounding-sam
    echo "✅ Activated grounding-sam environment"
else
    echo "⚠️  Conda not available, testing with current Python environment"
fi

# Test PyTorch CUDA availability
echo ""
echo "🧪 Testing PyTorch CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA is not available - PyTorch may need to be reinstalled with CUDA support')
" 2>/dev/null || echo "⚠️  PyTorch not installed in grounding-sam environment"

echo ""
echo "🎉 CUDA 11.8 configuration completed!"
echo ""
echo "📋 Summary:"
echo "  - CUDA_HOME: $CUDA_HOME"
echo "  - Environment variables configured for current session"
echo "  - Shell configuration files updated"
echo ""
echo "🔄 To apply changes in new terminal sessions:"
echo "  source ~/.bashrc"
echo "  # or restart your terminal"
