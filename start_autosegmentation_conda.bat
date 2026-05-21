@echo off
setlocal EnableExtensions

set "APP_DIR=%~dp0"
set "CONDA_ENV_NAME=yolo_stable"
set "CONDA_ENV_DIR="
set "EXE_FILE=%APP_DIR%AutoSegmentation.exe"

if not "%~1"=="" (
    set "CONDA_ENV_DIR=%~1"
) else if exist "%USERPROFILE%\miniconda3\envs\%CONDA_ENV_NAME%\python.exe" (
    set "CONDA_ENV_DIR=%USERPROFILE%\miniconda3\envs\%CONDA_ENV_NAME%"
) else if exist "%USERPROFILE%\anaconda3\envs\%CONDA_ENV_NAME%\python.exe" (
    set "CONDA_ENV_DIR=%USERPROFILE%\anaconda3\envs\%CONDA_ENV_NAME%"
) else if exist "C:\Users\innom\miniconda3\envs\%CONDA_ENV_NAME%\python.exe" (
    set "CONDA_ENV_DIR=C:\Users\innom\miniconda3\envs\%CONDA_ENV_NAME%"
)

echo [AutoSegmentation] Starting.
echo.

if not exist "%EXE_FILE%" (
    echo [ERROR] EXE file was not found.
    echo Path "%EXE_FILE%"
    pause
    exit /b 1
)

rem Do not activate conda before starting the PyInstaller exe.
rem The exe will use this Python only for training/auto-label/validation subprocesses.
if defined CONDA_ENV_DIR (
    if exist "%CONDA_ENV_DIR%\python.exe" (
        echo [AutoSegmentation] External Python:
        echo %CONDA_ENV_DIR%\python.exe
        set "AUTOSEGMENTATION_PYTHON_EXE=%CONDA_ENV_DIR%\python.exe"
        set "AUTOSEGMENTATION_CONDA_ENV_DIR=%CONDA_ENV_DIR%"
    ) else (
        echo [WARN] Requested conda environment was not found.
        echo Path "%CONDA_ENV_DIR%\python.exe"
        echo [WARN] Falling back to Python found by the application.
    )
) else (
    echo [AutoSegmentation] No yolo_stable conda environment found.
    echo [AutoSegmentation] Falling back to Python found by the application.
)
echo.

cd /d "%APP_DIR%"
"%EXE_FILE%"

if errorlevel 1 (
    echo.
    echo [ERROR] AutoSegmentation exited with error code %errorlevel%.
    pause
    exit /b %errorlevel%
)

endlocal
