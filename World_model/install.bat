@echo off
REM World4Omni Main Installation Script (Windows)
REM This script provides easy access to all installation utilities

set SCRIPT_DIR=%~dp0
set INSTALLATION_DIR=%SCRIPT_DIR%scripts\installation

echo 🚀 World4Omni Installation Manager
echo =================================
echo.
echo Available installation scripts:
echo   1. PyTorch with CUDA 11.8 installation
echo   2. CUDA 11.8 configuration
echo   3. Grounding-SAM environment verification
echo   4. Complete installation (recommended)
echo.

REM Check if installation directory exists
if not exist "%INSTALLATION_DIR%" (
    echo ❌ Installation scripts directory not found: %INSTALLATION_DIR%
    pause
    exit /b 1
)

REM Parse command line arguments
if "%1"=="pytorch" goto pytorch
if "%1"=="1" goto pytorch
if "%1"=="cuda" goto cuda
if "%1"=="2" goto cuda
if "%1"=="verify" goto verify
if "%1"=="3" goto verify
if "%1"=="all" goto complete
if "%1"=="4" goto complete
if "%1"=="complete" goto complete
if "%1"=="help" goto help
if "%1"=="-h" goto help
if "%1"=="--help" goto help
if "%1"=="" goto complete

echo ❌ Unknown option: %1
echo Run 'install.bat help' for usage information
pause
exit /b 1

:pytorch
echo 📦 Installing PyTorch with CUDA 11.8...
call "%INSTALLATION_DIR%\install_pytorch_cuda118.bat"
goto end

:cuda
echo 🔧 Configuring CUDA 11.8...
call "%INSTALLATION_DIR%\configure_cuda118.sh"
goto end

:verify
echo 🔍 Verifying Grounding-SAM environment...
call "%INSTALLATION_DIR%\verify_grounding_sam_env.sh"
goto end

:complete
echo 🎯 Running complete installation...
echo.
echo Step 1: Installing PyTorch with CUDA 11.8...
call "%INSTALLATION_DIR%\install_pytorch_cuda118.bat"
echo.
echo Step 2: Verifying installation...
call "%INSTALLATION_DIR%\verify_grounding_sam_env.sh"
goto end

:help
echo Usage: %0 [option]
echo.
echo Options:
echo   pytorch, 1    - Install PyTorch with CUDA 11.8
echo   cuda, 2       - Configure CUDA 11.8 environment
echo   verify, 3     - Verify Grounding-SAM environment
echo   all, 4        - Run complete installation (default)
echo   help, -h      - Show this help message
echo.
echo Examples:
echo   %0            # Run complete installation
echo   %0 pytorch    # Install PyTorch only
echo   %0 verify     # Verify environment only
goto end

:end
echo.
echo ✅ Installation process completed!
pause
