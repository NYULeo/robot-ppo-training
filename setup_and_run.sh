#!/bin/bash

# Robot PPO Training Setup and Run Script
# This script installs all requirements and runs the PPO training

echo "🤖 Robot PPO Training Setup and Run Script"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Requirements installed successfully!"
    else
        echo "❌ Failed to install requirements. Please check the requirements.txt file."
        exit 1
    fi
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Check if PPO.py exists
if [ ! -f "PPO.py" ]; then
    echo "❌ PPO.py not found!"
    exit 1
fi

echo ""
echo "🚀 Starting PPO training..."
echo "=========================="

# Run PPO.py
python3 PPO.py

# Deactivate virtual environment
deactivate

echo ""
echo "✅ Training completed!"
