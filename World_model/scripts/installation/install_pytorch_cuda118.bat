@echo off
REM World4Omni PyTorch Installation Script with CUDA 11.8 (Windows)
REM This script ensures consistent PyTorch installation with CUDA 11.8 support

echo 🚀 World4Omni PyTorch Installation Script (CUDA 11.8)
echo ==================================================

REM Check if conda is available
conda --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda is not installed or not in PATH
    pause
    exit /b 1
)

REM Activate grounding-sam environment
echo 🔧 Activating conda environment: grounding-sam
call conda activate grounding-sam

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

echo 📋 Python version:
python --version
echo 📋 Conda environment: grounding-sam

REM Set CUDA_HOME for CUDA 11.8
echo 🔧 Configuring CUDA_HOME for CUDA 11.8...
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set PATH=%CUDA_HOME%\bin;%PATH%
set LD_LIBRARY_PATH=%CUDA_HOME%\lib64;%LD_LIBRARY_PATH%

REM Uninstall existing PyTorch if present
echo 🧹 Uninstalling existing PyTorch installations...
pip uninstall -y torch torchvision torchaudio 2>nul

REM Install PyTorch with CUDA 11.8 support
echo 📦 Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

REM Verify installation
echo 🔍 Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available')"

REM Install project dependencies from requirements.txt
echo 📦 Installing project dependencies from requirements.txt...
pip install -r ..\..\requirements.txt

REM Install Grounded-SAM-2
echo 📦 Installing Grounded-SAM-2...
cd 3rdparty\Grounded-SAM-2
pip install -e ".[notebooks]"
cd ..\..

REM Test Grounded-SAM-2 installation
echo 🔍 Testing Grounded-SAM-2 installation...
python -c "import sam2; from sam2.build_sam import build_sam2; from grounding_dino.util.inference import load_model; print('✅ Grounded-SAM-2 imported successfully')"

echo.
echo 🎉 Installation completed successfully!
echo.
echo 📋 Summary:
python -c "import torch; print(f'  - PyTorch: {torch.__version__}'); print(f'  - CUDA Support: {torch.cuda.is_available()}')"
echo   - Grounded-SAM-2: Installed
echo.
echo 🚀 You can now run the World4Omni project!
echo.
echo Next steps:
echo   1. Run: python test_installation.py
echo   2. Check the INSTALLATION.md for additional setup instructions
echo   3. Run demo scripts in the scripts/ directory

pause
