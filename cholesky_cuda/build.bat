@echo off
REM Build the CUDA Cholesky program.
REM RTX 4050 Laptop = Ada Lovelace = sm_89.

nvcc -O3 -std=c++17 -arch=sm_89 -o cholesky_cuda.exe cholesky_cuda.cu

if %ERRORLEVEL% NEQ 0 (
    echo Build failed.
    exit /b %ERRORLEVEL%
)

echo Build succeeded: cholesky_cuda.exe
