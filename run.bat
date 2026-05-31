@echo off
title Mimic Chess AI Playground Launcher
color 0A

echo ===================================================================
echo             MIMIC CHESS AI PLAYGROUND - STARTUP SCRIPT
echo ===================================================================
echo.

:: Check for python launcher or python.exe
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=py
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set PYTHON_CMD=python
    ) else (
        color 0C
        echo [ERROR] Python is not installed or not in system PATH.
        echo Please install Python 3.8+ (x64) and check 'Add Python to environment variables'.
        pause
        exit /b 1
    )
)

echo [INFO] Using Python command: %PYTHON_CMD%

:: Create venv if it does not exist
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating Python virtual environment (venv)...
    %PYTHON_CMD% -m venv venv
    if %ERRORLEVEL% neq 0 (
        color 0C
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created successfully.
) else (
    echo [INFO] Virtual environment found.
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Checking and upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

echo [INFO] Installing standard packages (Flask, python-chess, NumPy)...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Failed to install standard packages.
    pause
    exit /b 1
)

:: Check if torch is installed
python -c "import torch" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing CPU-only PyTorch (lightweight, ~170MB)...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    if %ERRORLEVEL% neq 0 (
        color 0C
        echo [ERROR] Failed to install PyTorch.
        pause
        exit /b 1
    )
    echo [SUCCESS] PyTorch CPU installed.
) else (
    echo [INFO] PyTorch is already installed.
)

echo.
echo ===================================================================
echo    [SUCCESS] ALL DEPENDENCIES VERIFIED! STARTING SERVER...
echo    The Playground will be available at: http://127.0.0.1:5000
echo ===================================================================
echo.

:: Automatically open user's default browser in 2 seconds
start "" "http://127.0.0.1:5000"

:: Start the Flask app
python app.py
if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo [ERROR] Flask server stopped with an error.
    pause
)
