#!/bin/bash

# World4Omni PyTorch Installation Script with CUDA 11.8
# This script ensures consistent PyTorch installation with CUDA 11.8 support across the entire repository

set -e  # Exit on any error

echo "🚀 World4Omni PyTorch Installation Script (CUDA 11.8)"
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    exit 1
fi

# Activate grounding-sam environment
echo "🔧 Activating conda environment: grounding-sam"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate grounding-sam

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $PYTHON_VERSION"
echo "📋 Conda environment: grounding-sam"

if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l) -eq 1 ]]; then
    echo "⚠️  Warning: Python 3.10+ is recommended for optimal performance"
fi

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "📋 CUDA version: $CUDA_VERSION"
else
    echo "⚠️  Warning: CUDA toolkit not found. PyTorch will be installed with CUDA 11.8 support, but CUDA runtime may not be available."
fi

# Set CUDA_HOME for CUDA 11.8
echo "🔧 Configuring CUDA_HOME for CUDA 11.8..."
export CUDA_HOME=/usr/local/cuda-11.8

# Add CUDA_HOME to shell configuration if not already present
if ! grep -q "CUDA_HOME=/usr/local/cuda-11.8" ~/.bashrc 2>/dev/null; then
    echo "📝 Adding CUDA_HOME configuration to ~/.bashrc..."
    echo "" >> ~/.bashrc
    echo "# CUDA 11.8 Configuration for World4Omni" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda-11.8" >> ~/.bashrc
    echo "export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"" >> ~/.bashrc
    echo "✅ CUDA_HOME configuration added to ~/.bashrc"
else
    echo "✅ CUDA_HOME already configured in ~/.bashrc"
fi

# Uninstall existing PyTorch if present
echo "🧹 Uninstalling existing PyTorch installations..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch with CUDA 11.8 support
echo "📦 Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify installation
echo "🔍 Verifying PyTorch installation..."
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
    print('CUDA is not available - this may be expected if no GPU is present')
"

# Install project dependencies from requirements.txt
echo "📦 Installing project dependencies from requirements.txt..."
pip install -r ../../requirements.txt

# Install Grounded-SAM-2
echo "📦 Installing Grounded-SAM-2..."
cd 3rdparty/Grounded-SAM-2
pip install -e ".[notebooks]"
cd ../..

# Test Grounded-SAM-2 installation
echo "🔍 Testing Grounded-SAM-2 installation..."
python -c "
try:
    import sam2
    from sam2.build_sam import build_sam2
    from grounding_dino.util.inference import load_model
    print('✅ Grounded-SAM-2 imported successfully')
except ImportError as e:
    print(f'❌ Grounded-SAM-2 import failed: {e}')
    exit(1)
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Summary:"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA Support: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  - Grounded-SAM-2: Installed"
echo ""
echo "🚀 You can now run the World4Omni project!"
echo ""
echo "Next steps:"
echo "  1. Run: python test_installation.py"
echo "  2. Check the INSTALLATION.md for additional setup instructions"
echo "  3. Run demo scripts in the scripts/ directory"
