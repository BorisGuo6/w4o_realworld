# World4Omni Installation Guide

This guide provides step-by-step instructions for installing the World4Omni project and its dependencies.

## Table of Contents

- [CUDA 11.8 Consistency](#cuda-118-consistency)
- [Part 1: World Model Installation](#part-1-world-model-installation)
  - [1.1 Grounded-SAM-2 Setup](#11-grounded-sam-2-setup)
  - [1.2 Grounding DINO Setup](#12-grounding-dino-setup)
  - [1.3 SAM 2 Setup](#13-sam-2-setup)
- [Part 2: Additional Dependencies](#part-2-additional-dependencies)
- [Part 3: Verification](#part-3-verification)
- [Troubleshooting](#troubleshooting)

---

## CUDA 11.8 Consistency

This project is configured to use **CUDA 11.8** consistently across all components. This ensures:

- ✅ **Compatibility**: All PyTorch-based components work with the same CUDA version
- ✅ **Performance**: Optimal GPU acceleration across the entire pipeline
- ✅ **Stability**: Reduces version conflicts and runtime errors

### Quick Installation

For the fastest setup, use our automated installation script:

```bash
# Linux/macOS
./install_pytorch_cuda118.sh

# Windows
install_pytorch_cuda118.bat
```

This script will:
1. Install PyTorch with CUDA 11.8 support
2. Install all project dependencies
3. Set up Grounded-SAM-2
4. Verify the installation

### Manual Verification

To verify CUDA 11.8 is properly installed:

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Expected: 11.8')
"
```

### CUDA_HOME Configuration

Ensure CUDA_HOME is properly set to CUDA 11.8:

```bash
# Check current CUDA_HOME
echo "CUDA_HOME: $CUDA_HOME"

# Should show: /usr/local/cuda-11.8

# If not set correctly, run the configuration script:
./install.sh cuda

# Or set manually:
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

---

## Part 1: World Model Installation

### 1.1 Grounded-SAM-2 Setup

The Grounded-SAM-2 component is already included in the `3rdparty` directory. This section covers the installation and setup process.

#### Prerequisites

Before installing Grounded-SAM-2, ensure you have the following:

- **Operating System**: Linux (Ubuntu 18.04+ recommended)
- **Conda Environment**: `grounding-sam` (recommended)
- **Python**: ≥ 3.10
- **PyTorch**: Latest version with CUDA 11.8 support
- **CUDA**: 11.8 (recommended) or compatible version
- **GPU**: NVIDIA GPU with CUDA support

#### Environment Setup

**Option A: Use Existing Environment**
If you already have the `grounding-sam` conda environment:
```bash
conda activate grounding-sam
```

**Option B: Create New Environment**
If you need to create the environment:
```bash
conda create -n grounding-sam python=3.10 -y
conda activate grounding-sam
```

#### Step 1: Install PyTorch and CUDA

**Option A: Automated Installation (Recommended)**

Use our provided installation script to ensure consistent CUDA 11.8 setup:

```bash
# Linux/macOS (automatically activates grounding-sam environment)
./install.sh

# Windows (automatically activates grounding-sam environment)
install.bat
```

**Individual Scripts:**
```bash
# PyTorch installation only
./install.sh pytorch

# CUDA configuration only
./install.sh cuda

# Environment verification only
./install.sh verify
```

**Note**: The installation scripts will automatically activate the `grounding-sam` conda environment. If the environment doesn't exist, you'll need to create it first:

```bash
conda create -n grounding-sam python=3.10 -y
```

**Option B: Manual Installation**

If you prefer manual installation:

```bash
# Activate grounding-sam environment
conda activate grounding-sam

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

#### Step 2: Install Grounded-SAM-2

Navigate to the Grounded-SAM-2 directory and install:

```bash
cd /home/boris/workspace/World4Omni/3rdparty/Grounded-SAM-2

# Install Grounded-SAM-2 with all dependencies
pip install -e ".[notebooks]"
```

**Note**: If you encounter CUDA compilation issues, you can skip the CUDA extension build:

```bash
# Skip CUDA extension build (post-processing will be limited but functional)
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

#### Step 3: Verify Grounded-SAM-2 Installation

Test the installation:

```bash
cd /home/boris/workspace/World4Omni/3rdparty/Grounded-SAM-2

# Test basic import
python -c "import sam2; print('SAM 2 imported successfully')"

# Test Grounding DINO import
python -c "from grounding_dino.util.inference import load_model, load_image, predict, annotate; print('Grounding DINO imported successfully')"
```

### 1.2 Grounding DINO Setup

Grounding DINO is included as part of the Grounded-SAM-2 package. Additional setup may be required for specific models.

#### Download Pre-trained Models

```bash
cd /home/boris/workspace/World4Omni/3rdparty/Grounded-SAM-2

# Download Grounding DINO checkpoints
bash checkpoints/download_grounding_dino.sh

# Download Grounding DINO 1.5 checkpoints (optional)
bash gdino_checkpoints/download_gdino.sh
```

#### Verify Grounding DINO

```bash
# Test Grounding DINO functionality
python -c "
from grounding_dino.util.inference import load_model, load_image, predict
import torch

# Load model
model = load_model('grounding_dino/config/GroundingDINO_SwinT_OGC.py', 'checkpoints/groundingdino_swint_ogc.pth')
print('Grounding DINO model loaded successfully')
"
```

### 1.3 SAM 2 Setup

SAM 2 is automatically installed with Grounded-SAM-2. Additional configuration may be needed for optimal performance.

#### Download SAM 2 Checkpoints

```bash
cd /home/boris/workspace/World4Omni/3rdparty/Grounded-SAM-2

# Download SAM 2 checkpoints
python -c "
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# This will automatically download checkpoints when first used
print('SAM 2 checkpoints will be downloaded on first use')
"
```

#### Verify SAM 2 Installation

```bash
# Test SAM 2 functionality
python -c "
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Test model loading
sam2_checkpoint = 'checkpoints/sam2_hiera_large.pt'
model_cfg = 'sam2/configs/sam2.1/sam2.1_hiera_l.yaml'
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
print('SAM 2 model loaded successfully')
"
```

---

## Part 2: Additional Dependencies

Install the main project dependencies:

```bash
cd /home/boris/workspace/World4Omni

# Install main project requirements
pip install -r requirements.txt

# Install additional dependencies for world model
pip install opencv-python pycocotools supervision transformers timm
```

---

## Part 3: Verification

### Test Complete Installation

Create a test script to verify all components:

```bash
cd /home/boris/workspace/World4Omni

# Create test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify World4Omni installation
"""

def test_imports():
    """Test all critical imports"""
    try:
        # Test Grounded-SAM-2
        import sam2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✓ SAM 2 imported successfully")
        
        # Test Grounding DINO
        from grounding_dino.util.inference import load_model, load_image, predict
        print("✓ Grounding DINO imported successfully")
        
        # Test main project modules
        from utils.World_model import WorldModel
        print("✓ World4Omni modules imported successfully")
        
        # Test other dependencies
        import torch
        import cv2
        import numpy as np
        import transformers
        print("✓ All dependencies imported successfully")
        
        print("\n🎉 Installation verification completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
EOF

# Run test
python test_installation.py
```

### Test Grounded-SAM-2 Demo

Test with a sample image:

```bash
cd /home/boris/workspace/World4Omni/3rdparty/Grounded-SAM-2

# Run a simple demo (if you have a test image)
python grounded_sam2_local_demo.py --image_path /path/to/your/image.jpg --text_prompt "object"
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Compilation Errors

If you encounter CUDA compilation errors during installation:

```bash
# Option 1: Skip CUDA extension
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"

# Option 2: Set CUDA_HOME explicitly
export CUDA_HOME=/usr/local/cuda
pip install -e ".[notebooks]"
```

#### 2. PyTorch Version Conflicts

If you have PyTorch version conflicts:

```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision

# Reinstall with specific version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Missing Dependencies

If you encounter missing dependency errors:

```bash
# Install missing packages
pip install addict yapf timm supervision pycocotools

# For development dependencies
pip install -e ".[dev]"
```

#### 4. Model Download Issues

If model downloads fail:

```bash
# Set HuggingFace mirror (for Chinese users)
export HF_ENDPOINT=https://hf-mirror.com

# Or manually download models to checkpoints/ directory
```

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [Grounded-SAM-2 GitHub Issues](https://github.com/IDEA-Research/Grounded-SAM-2/issues)
2. Verify your CUDA and PyTorch versions are compatible
3. Ensure you have sufficient GPU memory (recommended: 8GB+)
4. Check the [SAM 2 Installation Guide](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)

---

## Next Steps

After completing Part 1 (World Model Installation):

1. **Part 2**: Install additional simulation dependencies (PyRep, RLBench)
2. **Part 3**: Configure environment variables and paths
3. **Part 4**: Run example scripts and demos

For the complete installation process, refer to the main project documentation.

---

*Last updated: January 2025*
