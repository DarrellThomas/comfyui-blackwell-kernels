@echo off
REM Build ComfyUI Blackwell Kernels on Windows
REM Requirements: CUDA 13.2 Toolkit, Python 3.10+, PyTorch 2.5+

echo === ComfyUI Blackwell Kernels — Build from Source ===
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    exit /b 1
)

REM Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo ERROR: PyTorch not found. Install with:
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu132
    exit /b 1
)

REM Check CUDA
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; cc=torch.cuda.get_device_capability(); print(f'GPU: CC {cc[0]}.{cc[1]}'); assert cc>=(12,0), f'Need CC 12.0+, got {cc[0]}.{cc[1]}'" 2>nul
if errorlevel 1 (
    echo ERROR: No RTX 5090 detected (need Compute Capability 12.0+)
    exit /b 1
)

REM Check CUDA toolkit
where nvcc >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" (
        set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
        set "PATH=%CUDA_HOME%\bin;%PATH%"
    ) else (
        echo ERROR: CUDA Toolkit 13.2 not found. Download from:
        echo   https://developer.nvidia.com/cuda-13-2-0-download-archive
        exit /b 1
    )
)

echo.
echo Building CUDA kernels (this takes a few minutes)...
set TORCH_CUDA_ARCH_LIST=12.0a
pip install -e . 2>&1
if errorlevel 1 (
    echo.
    echo BUILD FAILED. Common fixes:
    echo   1. Install Visual Studio Build Tools (C++ workload)
    echo   2. Set CUDA_HOME to your CUDA 13.2 install path
    echo   3. Make sure PyTorch CUDA version matches toolkit
    exit /b 1
)

echo.
echo === BUILD SUCCESSFUL ===
echo.
echo To use: copy this folder into ComfyUI\custom_nodes\
echo The kernels activate automatically on RTX 5090.
