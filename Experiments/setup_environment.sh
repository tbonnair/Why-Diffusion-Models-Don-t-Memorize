#!/bin/bash

# Setup script for Diffusion Models Environment
# This script creates a conda environment for running the diffusion models code

echo "========================================"
echo "Diffusion Models Environment Setup"
echo "========================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Ask user for GPU or CPU version
echo "Select installation type:"
echo "  1) GPU version (requires CUDA-compatible GPU)"
echo "  2) CPU-only version"
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    ENV_FILE="environment.yml"
    echo ""
    echo "Installing GPU version..."
    echo "Note: Make sure you have CUDA 11.8 or 12.1 installed"
    echo "      Edit environment.yml to match your CUDA version if needed"
elif [ "$choice" = "2" ]; then
    ENV_FILE="environment_cpu.yml"
    echo ""
    echo "Installing CPU-only version..."
else
    echo "❌ Invalid choice. Exiting."
    exit 1
fi

echo ""
echo "Creating conda environment from $ENV_FILE..."
echo ""

# Create the environment
conda env create -f "$ENV_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Environment created successfully!"
    echo "========================================"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate diffusion-models"
    echo ""
    echo "To test the installation:"
    echo "  conda activate diffusion-models"
    echo "  python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
    echo ""
    echo "To run the code:"
    echo "  cd Experiments/src/Training"
    echo "  python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1"
else
    echo ""
    echo "❌ Error: Failed to create environment"
    echo "Please check the error messages above"
    exit 1
fi
