#!/bin/bash

# GitHub Repository Setup Script
# This script helps you set up the GitHub repository

echo "🚀 GitHub Repository Setup"
echo "=========================="

echo ""
echo "📋 Instructions to create and push to GitHub:"
echo ""
echo "1. Go to https://github.com/new"
echo "2. Create a new repository named 'robot-ppo-training' (or your preferred name)"
echo "3. DO NOT initialize with README, .gitignore, or license (we already have these)"
echo "4. Copy the repository URL (it will look like: https://github.com/YOUR_USERNAME/robot-ppo-training.git)"
echo ""
echo "5. Then run the following commands (replace YOUR_USERNAME and REPO_NAME):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "6. Or if you want to use SSH (if you have SSH keys set up):"
echo "   git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""

# Check if remote is already set
if git remote -v | grep -q origin; then
    echo "✅ Remote origin is already configured:"
    git remote -v
    echo ""
    echo "To push to GitHub, run:"
    echo "   git push -u origin main"
else
    echo "❌ No remote origin configured yet."
    echo "Follow the instructions above to set up the remote."
fi

echo ""
echo "🎯 After pushing to GitHub, you can run the training script:"
echo "   ./setup_and_run.sh  (on macOS/Linux)"
echo "   setup_and_run.bat   (on Windows)"
