#!/bin/bash

# Grounding-SAM Environment Verification Script
# This script verifies the grounding-sam conda environment setup

set -e

echo "🔍 Grounding-SAM Environment Verification"
echo "========================================"

# Activate grounding-sam environment
echo "🔧 Activating grounding-sam environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate grounding-sam

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $PYTHON_VERSION"

# Check PyTorch installation
echo ""
echo "🧪 Testing PyTorch installation..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA is not available')
"

# Check CUDA_HOME
echo ""
echo "🔧 Checking CUDA_HOME configuration..."
echo "CUDA_HOME: $CUDA_HOME"
if [ "$CUDA_HOME" = "/usr/local/cuda-11.8" ]; then
    echo "✅ CUDA_HOME correctly set to CUDA 11.8"
else
    echo "⚠️  CUDA_HOME not set to CUDA 11.8"
    echo "   Run: export CUDA_HOME=/usr/local/cuda-11.8"
fi

# Test project dependencies from requirements.txt
echo ""
echo "🧪 Testing project dependencies..."
python -c "
import sys
dependencies = [
    'opencv-python', 'pycocotools', 'supervision', 'numpy', 
    'transformers', 'yapf', 'timm', 'gymnasium'
]

for dep in dependencies:
    try:
        if dep == 'opencv-python':
            import cv2
            print(f'✅ {dep} (cv2) imported successfully')
        elif dep == 'pycocotools':
            import pycocotools
            print(f'✅ {dep} imported successfully')
        elif dep == 'supervision':
            import supervision
            print(f'✅ {dep} imported successfully')
        elif dep == 'numpy':
            import numpy
            print(f'✅ {dep} imported successfully')
        elif dep == 'transformers':
            import transformers
            print(f'✅ {dep} imported successfully')
        elif dep == 'yapf':
            import yapf
            print(f'✅ {dep} imported successfully')
        elif dep == 'timm':
            import timm
            print(f'✅ {dep} imported successfully')
        elif dep == 'gymnasium':
            import gymnasium
            print(f'✅ {dep} imported successfully')
    except ImportError as e:
        print(f'❌ {dep} import failed: {e}')
"

# Test Grounded-SAM-2 imports
echo ""
echo "🧪 Testing Grounded-SAM-2 imports..."
python -c "
try:
    import sam2
    print('✅ SAM 2 imported successfully')
except ImportError as e:
    print(f'❌ SAM 2 import failed: {e}')

try:
    from grounding_dino.util.inference import load_model
    print('✅ Grounding DINO imported successfully')
except ImportError as e:
    print(f'❌ Grounding DINO import failed: {e}')
"

# Check if Grounded-SAM-2 is installed
echo ""
echo "🔍 Checking Grounded-SAM-2 installation..."
if [ -d "3rdparty/Grounded-SAM-2" ]; then
    echo "✅ Grounded-SAM-2 directory found"
    cd 3rdparty/Grounded-SAM-2
    if pip show SAM-2 >/dev/null 2>&1; then
        echo "✅ SAM-2 package installed"
    else
        echo "⚠️  SAM-2 package not installed"
        echo "   Run: pip install -e ."
    fi
    cd ../..
else
    echo "❌ Grounded-SAM-2 directory not found"
fi

echo ""
echo "🎉 Verification completed!"
echo ""
echo "📋 Summary:"
echo "  - Environment: grounding-sam"
echo "  - Python: $PYTHON_VERSION"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA Support: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  - CUDA_HOME: $CUDA_HOME"
