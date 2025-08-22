@echo off
REM Robot PPO Training Setup and Run Script for Windows
REM This script installs all requirements and runs the PPO training

echo 🤖 Robot PPO Training Setup and Run Script
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo 🐍 Python version: %python_version%

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing requirements...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements. Please check the requirements.txt file.
        pause
        exit /b 1
    ) else (
        echo ✅ Requirements installed successfully!
    )
) else (
    echo ❌ requirements.txt not found!
    pause
    exit /b 1
)

REM Check if PPO.py exists
if not exist "PPO.py" (
    echo ❌ PPO.py not found!
    pause
    exit /b 1
)

echo.
echo 🚀 Starting PPO training...
echo ==========================

REM Run PPO.py
python PPO.py

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
echo ✅ Training completed!
pause
