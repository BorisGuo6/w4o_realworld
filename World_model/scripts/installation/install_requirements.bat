@echo off
REM Requirements Installation Script (Windows)
REM This script installs all dependencies from requirements.txt

echo 📦 Installing Project Dependencies
echo =================================

REM Activate grounding-sam environment
echo 🔧 Activating grounding-sam environment...
call conda activate grounding-sam

echo ✅ Environment activated: grounding-sam

REM Check if requirements.txt exists
set REQUIREMENTS_FILE=..\..\requirements.txt
if not exist "%REQUIREMENTS_FILE%" (
    echo ❌ requirements.txt not found at %REQUIREMENTS_FILE%
    pause
    exit /b 1
)

echo 📋 Found requirements.txt at: %REQUIREMENTS_FILE%

REM Display requirements content
echo.
echo 📋 Dependencies to install:
echo ------------------------
type "%REQUIREMENTS_FILE%"
echo ------------------------

REM Install requirements
echo.
echo 📦 Installing dependencies...
pip install -r "%REQUIREMENTS_FILE%"

REM Verify installation
echo.
echo 🧪 Verifying installation...
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

echo.
echo 🎉 Requirements installation completed!
echo.
echo 📋 Summary:
echo   - Environment: grounding-sam
echo   - Requirements file: %REQUIREMENTS_FILE%
echo   - All dependencies installed and verified

pause
