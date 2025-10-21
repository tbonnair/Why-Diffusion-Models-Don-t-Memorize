# Environment Setup for training diffusion models

This repository contains code for the numerical experiments carried out in the paper [Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training](https://arxiv.org/abs/2505.17638) by T. Bonnaire, R. Urfin, G. Biroli and M. Mézard.

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

### Option 3: Manual (with pip)
Sometimes, conda environment creation from `yml` can be very slow.
You can alternatively install packages in an environment using the following pip command:
```bash
pip install -r requirements.txt
```

## Activating the Environment
```bash
conda activate memorization
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
The models and generated images along training will be stored in `Experiments/Saves/CelebA32_1024_32_Adam_512_0.0001_index0`.

Parameters:
- `-n`: Number of training images (1024)
- `-i`: Dataset index: training images will be those indexed from i*n to (i+1)*n
- `-s`: Image size (32)
- `-LR`: Learning rate (0.0001)
- `-O`: Optimizer (Adam or SGD_Momentum)
- `-W`: Number of base filters (32)
- `-t`: Time step (-1 for normal mode, integer in [0, T-1] for fixed-time training)

### Example 3: Generation with trained U-Net models

The `generate.py` script allows you to generate samples from trained diffusion models at various training checkpoints. It loads a pre-trained U-Net diffusion model and generates synthetic images using the DDIM (Denoising Diffusion Implicit Models) sampling method.
Example:
```bash
cd Experiments/src/Generation
python generate.py -D CelebA -n 1024 -i 0 -s 32 -B 512 -LR 0.0001 -O Adam -W 32 -Ns 100 --device cuda:0
```
It will create a folder `Samples` in `Experiments/Saves/CelebA32_1024_32_Adam_512_0.0001_index0` with multiples subfolders corresponding to the several snapshot of trained models.
To modify these generation times, you can modify `generate.py`, making sure it fits the models saved in `run_Unet.py` as well.

Parameters:
- `-D`: Dataset (CelebA)
- `-n`: Number of training images (1024)
- `-i`: Dataset index
- `-s`: Image size (32)
- `-B`: Batch size used to train the model (512)
- `-LR`: Learning rate (0.0001)
- `-O`: Optimizer (Adam or SGD_Momentum)
- `-W`: Number of base filters (32)
- `-Ns`: Number of samples to generate
- `--device`: device to use (default is cuda:0)

### Example 4: Computing Memorization fraction

The `compute_fmem.py` script computes the fraction of generated samples that are in fact memorizing the training data points. It analyzes the k-nearest neighbor distances using a gap ratio analysis.

Example:
```bash
cd Experiments/src/Evaluation
python compute_fmem.py -D CelebA -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 -Ns 1 --gap_threshold 0.333 --device cuda:0
```

This will analyze the generated samples from the corresponding model and save memorization metrics to `Experiments/Saves/CelebA32_1024_32_Adam_512_0.0001_index0/Memorization/fraction_collapse.txt`.

Parameters:
- `-D`: Dataset (CelebA)
- `-n`: Number of training images (1024)
- `-i`: Dataset index
- `-s`: Image size (32)
- `-LR`: Learning rate (0.0001)
- `-O`: Optimizer (Adam or SGD_Momentum)
- `-W`: Number of base filters (32)
- `-B`: Batch size (512)
- `-Ns`: Number of sample batches to analyze (basically Ns used in generation divided by 100).
- `--batch_sample_size`: Size of each sample batch (default: 100)
- `--gap_threshold`: Gap ratio threshold for collapsed samples (default: 1/3 ≈ 0.333)
- `--device`: Device to use (cuda:0, cpu)
- `--plots`: Generate diagnostic plots (optional flag)

**Note**: This script requires that you have already generated samples using `generate.py` (Example 3) for the same model configuration.

### Example 5: Computing FID (Fréchet Inception Distance)

The `compute_FID.py` script computes the FID score between generated samples and reference statistics. FID measures the quality and diversity of generated images by comparing feature distributions in the Inception network's feature space. Lower FID scores indicate better generative quality.

Example:
```bash
cd Experiments/src/Evaluation
python compute_FID.py -D CelebA -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 -istat 1 --N1 0 --N2 1 --device cuda:0
```

This will compute FID scores for all training checkpoints and save results to `Experiments/Saves/FID/CelebA32_1024_32_Adam_512_0.0001_index0/FID_1.txt`.

Parameters:
- `-D`: Dataset (CelebA)
- `-n`: Number of training images (1024)
- `-i`: Dataset index
- `-s`: Image size (32)
- `-LR`: Learning rate (0.0001)
- `-O`: Optimizer (Adam or SGD_Momentum)
- `-W`: Number of base filters (32)
- `-B`: Batch size (512)
- `-istat`: Index of reference statistics file (1-5, corresponds to `stats1.npz` through `stats5.npz` in the `Experiments/Saves/FID_ref`folder.)
- `--N1`: Starting batch index for analysis (default: 0)
- `--N2`: Ending batch index for analysis (default: 100)
- `--batch_size_samples`: Size of each generated sample batch (default: 100)
- `--device`: Device to use (cuda:0, cpu)

**Prerequisites**: 
1. Generated samples from `generate.py` (Example 3)
2. Reference statistics file (`../Saves/FID/stats{istat}.npz`) must exist
3. `pytorch-fid` package installed (`pip install pytorch-fid`)

**Note**: The script temporarily saves images as PNG files for FID computation and automatically cleans them up afterward to save disk space.

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
All the code above was tested successfully on NVIDIA RTX 2080 Ti GPUs (12 GB memory).
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
- pytorch-fid >= 0.3.0