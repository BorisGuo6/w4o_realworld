# Installation Scripts

This directory contains all installation and configuration scripts for the World4Omni project.

## Scripts Overview

### Core Installation Scripts

| Script | Platform | Description |
|--------|----------|-------------|
| `install_pytorch_cuda118.sh` | Linux/macOS | Installs PyTorch with CUDA 11.8 support in grounding-sam environment |
| `install_pytorch_cuda118.bat` | Windows | Windows version of PyTorch installation |
| `configure_cuda118.sh` | Linux/macOS | Configures CUDA_HOME and environment variables for CUDA 11.8 |
| `verify_grounding_sam_env.sh` | Linux/macOS | Verifies grounding-sam environment setup |

### Quick Start

**From project root directory:**

```bash
# Run complete installation (recommended)
./install.sh

# Or run individual scripts
./install.sh pytorch    # Install PyTorch only
./install.sh cuda       # Configure CUDA only
./install.sh verify     # Verify environment only
```

**Windows:**
```cmd
install.bat
install.bat pytorch
install.bat cuda
install.bat verify
```

### Individual Script Usage

#### 1. PyTorch Installation

**Linux/macOS:**
```bash
cd scripts/installation
./install_pytorch_cuda118.sh
```

**Windows:**
```cmd
cd scripts\installation
install_pytorch_cuda118.bat
```

**What it does:**
- Activates `grounding-sam` conda environment
- Uninstalls existing PyTorch installations
- Installs PyTorch with CUDA 11.8 support
- Installs project dependencies from `requirements.txt`
- Installs Grounded-SAM-2
- Verifies installation

#### 2. CUDA Configuration

**Linux/macOS:**
```bash
cd scripts/installation
./configure_cuda118.sh
```

**What it does:**
- Sets CUDA_HOME to `/usr/local/cuda-11.8`
- Updates shell configuration files (.bashrc, .profile)
- Tests CUDA availability
- Verifies PyTorch CUDA support

#### 3. Environment Verification

**Linux/macOS:**
```bash
cd scripts/installation
./verify_grounding_sam_env.sh
```

**What it does:**
- Activates `grounding-sam` environment
- Checks Python version
- Tests PyTorch installation and CUDA support
- Verifies CUDA_HOME configuration
- Tests Grounded-SAM-2 imports

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS, or Windows
- **Conda**: Anaconda or Miniconda installed
- **CUDA**: 11.8 toolkit installed
- **GPU**: NVIDIA GPU with CUDA support

### Environment Setup
- **Conda Environment**: `grounding-sam` (will be created if not exists)
- **Python**: ≥ 3.10

### Project Dependencies

The installation scripts automatically install all dependencies from `requirements.txt`:

```
# PyTorch with CUDA 11.8 support
torch>=2.3.1
torchvision>=0.18.1

# Other dependencies
opencv-python
pycocotools
supervision
numpy<2
transformers
yapf
timm
gymnasium
```

**Manual installation:**
```bash
conda activate grounding-sam
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

1. **Conda environment not found**
   ```bash
   conda create -n grounding-sam python=3.10 -y
   ```

2. **CUDA_HOME not set**
   ```bash
   export CUDA_HOME=/usr/local/cuda-11.8
   ```

3. **PyTorch CUDA not available**
   ```bash
   # Reinstall PyTorch with CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Permission denied on scripts**
   ```bash
   chmod +x *.sh
   ```

### Getting Help

If you encounter issues:

1. Check the main [INSTALLATION.md](../../INSTALLATION.md) file
2. Run the verification script: `./verify_grounding_sam_env.sh`
3. Check conda environment: `conda info --envs`
4. Verify CUDA installation: `nvcc --version`

## File Structure

```
scripts/installation/
├── README.md                           # This file
├── install_pytorch_cuda118.sh         # PyTorch installation (Linux/macOS)
├── install_pytorch_cuda118.bat        # PyTorch installation (Windows)
├── configure_cuda118.sh               # CUDA configuration
└── verify_grounding_sam_env.sh        # Environment verification
```

## Notes

- All scripts are designed to work with the `grounding-sam` conda environment
- Scripts automatically handle environment activation
- CUDA 11.8 is required for optimal performance
- Scripts include error handling and verification steps
