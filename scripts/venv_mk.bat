@echo off
REM Check if an argument was provided
IF "%~1"=="" (
    echo Checking NVIDIA GPU availability...
    nvidia-smi >nul 2>&1

    IF %ERRORLEVEL% NEQ 0 (
        echo [FAIL] nvidia-smi is not available or failed to run.
        echo Running uv sync with CPU only support...
        uv sync --extra cpu
    ) ELSE (
        echo [PASS] nvidia-smi works correctly.
        echo Running uv sync with GPU support...
        uv sync --extra gpu
    )
) ELSE (
    REM Run uv sync with the provided argument
    uv sync --extra %~1
)

pause
