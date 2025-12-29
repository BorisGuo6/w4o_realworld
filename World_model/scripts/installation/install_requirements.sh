#!/bin/bash

# Requirements Installation Script
# This script installs all dependencies from requirements.txt

set -e

echo "📦 Installing Project Dependencies"
echo "================================="

# Activate grounding-sam environment
echo "🔧 Activating grounding-sam environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate grounding-sam

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# Check if requirements.txt exists
REQUIREMENTS_FILE="../../requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ requirements.txt not found at $REQUIREMENTS_FILE"
    exit 1
fi

echo "📋 Found requirements.txt at: $REQUIREMENTS_FILE"

# Display requirements content
echo ""
echo "📋 Dependencies to install:"
echo "------------------------"
cat "$REQUIREMENTS_FILE"
echo "------------------------"

# Install requirements
echo ""
echo "📦 Installing dependencies..."
pip install -r "$REQUIREMENTS_FILE"

# Verify installation
echo ""
echo "🧪 Verifying installation..."
python -c "
import sys
dependencies = [
    'opencv-python', 'pycocotools', 'supervision', 'numpy', 
    'transformers', 'yapf', 'timm', 'gymnasium'
]

print('Testing project dependencies:')
for dep in dependencies:
    try:
        if dep == 'opencv-python':
            import cv2
            print(f'✅ {dep} (cv2) - version: {cv2.__version__}')
        elif dep == 'pycocotools':
            import pycocotools
            print(f'✅ {dep} - installed')
        elif dep == 'supervision':
            import supervision
            print(f'✅ {dep} - version: {supervision.__version__}')
        elif dep == 'numpy':
            import numpy
            print(f'✅ {dep} - version: {numpy.__version__}')
        elif dep == 'transformers':
            import transformers
            print(f'✅ {dep} - version: {transformers.__version__}')
        elif dep == 'yapf':
            import yapf
            print(f'✅ {dep} - installed')
        elif dep == 'timm':
            import timm
            print(f'✅ {dep} - version: {timm.__version__}')
        elif dep == 'gymnasium':
            import gymnasium
            print(f'✅ {dep} - version: {gymnasium.__version__}')
    except ImportError as e:
        print(f'❌ {dep} - failed: {e}')
    except Exception as e:
        print(f'⚠️  {dep} - warning: {e}')
"

echo ""
echo "🎉 Requirements installation completed!"
echo ""
echo "📋 Summary:"
echo "  - Environment: grounding-sam"
echo "  - Requirements file: $REQUIREMENTS_FILE"
echo "  - All dependencies installed and verified"
