#!/bin/bash

# GPU Server Setup Script for H780 Robot Training
# This script is optimized for remote server deployment with GPU support

echo "🤖 H780 Robot GPU Server Setup Script"
echo "====================================="

# Check if running on a remote server
if [ -z "$DISPLAY" ] || [ "$DISPLAY" = ":0" ]; then
    echo "🖥️  Detected headless/remote server environment"
    export HEADLESS_MODE=true
else
    echo "🖥️  Detected local environment with display"
    export HEADLESS_MODE=false
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $python_version"

# Check for NVIDIA GPU
echo "🔍 Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    export CUDA_AVAILABLE=true
else
    echo "⚠️  NVIDIA GPU not detected or drivers not installed"
    echo "   Training will use CPU (slower but functional)"
    export CUDA_AVAILABLE=false
fi

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

# Install PyTorch with CUDA support if available
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🚀 Installing PyTorch with CUDA support..."
    
    # Detect CUDA version
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "Detected CUDA version: $cuda_version"
    
    # Install appropriate PyTorch version
    if [[ "$cuda_version" == "11.8" ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$cuda_version" == "12.1" ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "⚠️  Unknown CUDA version, installing CPU version"
        pip install torch torchvision
    fi
else
    echo "💻 Installing PyTorch CPU version..."
    pip install torch torchvision
fi

# Install other requirements
echo "📚 Installing other requirements..."
if [ -f "requirements_gpu.txt" ]; then
    pip install -r requirements_gpu.txt
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ No requirements file found!"
    exit 1
fi

# Verify PyTorch CUDA installation
echo "🔍 Verifying PyTorch CUDA installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA not available - will use CPU')
"

# Test environment creation
echo "🧪 Testing environment creation..."
python3 -c "
try:
    from H780_Train import H780MuJoCoEnv
    env = H780MuJoCoEnv(
        urdf_path='H780bv2.SLDASM/H780bv2.SLDASM.xml',
        render_mode=None,
        enable_animation=False
    )
    obs, info = env.reset()
    print(f'✅ Environment created successfully! Observation shape: {obs.shape}')
    env.close()
except Exception as e:
    print(f'❌ Environment creation failed: {e}')
    print('Please check the diagnostic script for more details')
"

# Set environment variables for GPU training
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🔧 Setting GPU environment variables..."
    echo "export CUDA_VISIBLE_DEVICES=0" >> venv/bin/activate
    echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" >> venv/bin/activate
    echo "✅ GPU environment variables configured"
fi

# Create a GPU training script
echo "📝 Creating GPU training script..."
cat > train_gpu.py << 'EOF'
#!/usr/bin/env python3
"""
GPU-optimized training script for H780 robot
"""

import os
import torch
from PPO import PPO
from H780_Train import H780MuJoCoEnv

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create environment
    env = H780MuJoCoEnv(
        urdf_path="H780bv2.SLDASM/H780bv2.SLDASM.xml",
        render_mode=None,  # Disable rendering for headless servers
        enable_animation=False
    )
    
    # Initialize PPO
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_ratio=0.2,
        epochs=4,
        batch_size=32,
        max_timesteps_per_episode=1000
    )
    
    # Start training
    print("🚀 Starting GPU training...")
    ppo.train(env, max_episodes=1000)

if __name__ == "__main__":
    main()
EOF

chmod +x train_gpu.py

echo ""
echo "✅ GPU server setup complete!"
echo ""
echo "To start training:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run GPU training: python train_gpu.py"
echo ""
echo "To run diagnostics:"
echo "python gpu_diagnostic.py"
echo ""
echo "For troubleshooting, check the diagnostic script output."

# Deactivate virtual environment
deactivate
