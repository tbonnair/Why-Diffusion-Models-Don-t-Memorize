# Environment Setup for Diffusion Models

This repository contains code for the numerical experiments carried out in the paper "Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training" by T. Bonnaire, R. Urfin, G. Biroli and M. Mézard.

## Quick Setup

### Option 1: Automated Setup (Recommended)

Run the setup script:

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

The script will guide you through:
1. Checking if conda is installed
2. Choosing between GPU or CPU version
3. Creating the environment
4. Providing activation instructions

### Option 2: Manual Setup

#### For GPU (CUDA 11.8):
```bash
conda env create -f environment.yml
```

#### For CPU-only:
```bash
conda env create -f environment_cpu.yml
```

#### For CUDA 12.x:
Edit `environment.yml` and change:
```yaml
- pytorch-cuda=11.8
```
to:
```yaml
- pytorch-cuda=12.1
```
Then run:
```bash
conda env create -f environment.yml
```

## Activating the Environment

```bash
conda activate diffusion-models
```

## Running the Code

### Example 1: GMM Training
```bash
cd Experiments/src/Training
python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1
```

Parameters:
- `-n`: Number of training samples (4096)
- `-d`: Dimension (8)
- `-s`: Random seed (1)
- `-de`: Network width (128)
- `-O`: Optimizer (Adam or SGD_Momentum)
- `-B`: Batch size (512)
- `-t`: Time step (-1 for normal mode, integer in [0, T-1] for fixed-time training)

### Example 2: CelebA Training
```bash
cd Experiments/src/Training
python run_Unet.py -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -t -1
```

Parameters:
- `-n`: Number of training images (1024)
- `-i`: Dataset index: training images will be those indexed from i*n to (i+1)*n
- `-s`: Image size (32)
- `-LR`: Learning rate (0.0001)
- `-O`: Optimizer (Adam or SGD_Momentum)
- `-W`: Number of base filters (32)
- `-t`: Time step (-1 for normal mode, integer in [0, T-1] for fixed-time training)

### Generation with trained U-Net models

The `generate.py` script allows you to generate samples from trained diffusion models at various training checkpoints. It loads a pre-trained U-Net diffusion model and generates synthetic images using the DDIM (Denoising Diffusion Implicit Models) sampling method.
Example:
```bash
cd Experiments/src/Generation
python generate.py -D CelebA -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -Ns 100 --device cuda:0
```

Parameters:
- `-D`: Dataset (CelebA)
- `-n`: Number of training images (1024)
- `-i`: Dataset index
- `-s`: Image size (32)
- `-LR`: Learning rate (0.0001)
- `-O`: Optimizer (Adam or SGD_Momentum)
- `-W`: Number of base filters (32)
- `-Ns`: Number of samples to generate
- `--device`: device to use (default is cuda:0)



## Updating the Environment

If you need to install additional packages:

```bash
conda activate diffusion-models
conda install <package-name>
# or
pip install <package-name>
```

To update the environment file after adding packages:

```bash
conda env export > environment_updated.yml
```

## Troubleshooting

### CUDA Version Mismatch
If you get CUDA errors, check your CUDA version:
```bash
nvidia-smi
```
Then edit `environment.yml` to match your CUDA version.

### Import Errors
Make sure you're in the correct directory when running scripts. The code uses relative imports, so run from:
- `Experiments/src/Training/` for training scripts
- `Experiments/src/Evaluation/` for evaluation scripts
- `Experiments/src/Generation/` for generation scripts

### Memory Issues
If you run out of GPU memory:
- Reduce batch size (`-B` parameter)
- Reduce image size (`-s` parameter for CelebA)
- Reduce network width (`-W` parameter)

## Dependencies

Main packages included:
- PyTorch >= 2.0.0 (with CUDA 11.8 or CPU)
- torchvision
- numpy >= 1.23.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0
- tqdm >= 4.65.0
- Pillow >= 9.0.0
- natsort

## System Requirements

### Minimum (CPU-only):
- 8 GB RAM
- 10 GB disk space

### Recommended (GPU):
- NVIDIA GPU with 8+ GB VRAM (RTX 3070 or better)
- 16+ GB RAM
- 20+ GB disk space
- CUDA 11.8 or 12.1
