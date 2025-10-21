"""
Compute fraction collapsed (memorization metric) for diffusion models.
This script analyzes generated samples to compute the fraction of samples that collapse
to training data using gap ratio analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import warnings

# Add Utils to path
sys.path.insert(1, '../Utils/')      # In case we run from Experiments/Evaluation
import Diffusion as dm
import cfg

warnings.filterwarnings("ignore")


def bootstrap_mean_se(data, threshold, n_bootstrap=1000, random_state=None):
    """
    Compute bootstrap estimate of the mean and its standard error for values below a threshold.

    Parameters:
    - data: 1D array-like of values.
    - threshold: numeric threshold; only values < threshold are considered.
    - n_bootstrap: number of bootstrap samples.
    - random_state: seed for reproducibility.

    Returns:
    - mean_est: bootstrap estimate of the mean.
    - se_est: bootstrap estimate of the standard error of the mean.
    - lower: lower bound of 95% confidence interval.
    - upper: upper bound of 95% confidence interval.
    """
    # Prepare RNG
    rng = np.random.default_rng(random_state)
    
    # Generate bootstrap samples
    means = np.empty(n_bootstrap)
    n_data = len(data)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n_data, replace=True)
        collapsed = np.where(sample < threshold)[0]
        means[i] = len(collapsed) / len(sample)
    
    # Compute estimates
    mean_est = means.mean()
    se_est = means.std(ddof=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return mean_est, se_est, lower, upper


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute fraction collapsed (memorization metric) for diffusion models."
    )
    
    # Model configuration arguments
    parser.add_argument("-n", "--num", help="Number of training data", type=int, required=True)
    parser.add_argument("-i", "--index", help="Index for the dataset (0 or 1)", type=int, required=True)
    parser.add_argument("-s", "--img_size", help="Size of the images to use", type=int, required=True)
    parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float, required=True)
    parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str, required=True)
    parser.add_argument("-W", "--nbase", help="Number of base filters", type=int, required=True)
    parser.add_argument("-B", "--batch_size", help="Batch size used to train the model", type=int, required=True)
    parser.add_argument("-D", "--dataset", help="Dataset used to train the model", type=str, required=True)
    
    # Analysis parameters
    parser.add_argument("-Ns", "--Nsamples", help="Number of sample batches to analyze", type=int, default=100)
    parser.add_argument("--batch_sample_size", help="Size of each sample batch", type=int, default=100)
    parser.add_argument("--gap_threshold", help="Gap ratio threshold for collapsed samples", type=float, default=1/3)
    parser.add_argument("--device", help="Device to use (cuda:0, cpu)", type=str, default='cuda:0')
    
    return parser.parse_args()

def compute_fraction_mem(training_times, train_images, type_model, config, file_fc,
                             nsamples, sample_size, gap_threshold):
    """Compute fraction collapsed for all training times."""
    N = np.prod(config.IMG_SHAPE)
    X = train_images.reshape(-1, N).float()
    
    pbar = tqdm(training_times)
    for tau in pbar:
        # Load generated images and compute k-nearest neighbors
        k = min(2, len(train_images))
        distances_tensor_all = torch.zeros(nsamples * sample_size, k)
        knn_tensor_all = torch.zeros(nsamples * sample_size, k)
        
        for i in range(nsamples):
            path_save = config.path_save + type_model + 'Samples/' + '/{:d}/'.format(tau)
            path = path_save + 'generated'
            file_a = path + '/samples_a_{:d}'.format(i)
            
            try:
                images_a = torch.load(file_a)
            except FileNotFoundError:
                print(f"Warning: File not found: {file_a}")
                continue
            
            i1, i2 = i * sample_size, (i + 1) * sample_size
            
            # Compute distances to training set
            s = images_a.reshape(-1, 1, N).to(config.DEVICE)
            dist = torch.norm(s - X, dim=2, p=2)
            knn = dist.topk(k, dim=1, largest=False)
            
            distances_tensor_all[i1:i2, :] = knn[0].cpu()
            knn_tensor_all[i1:i2, :] = knn[1].cpu()
        
        # Compute gap ratios
        gap_ratio = distances_tensor_all[:, 0] / distances_tensor_all[:, 1]
        
        # Compute fraction collapsed with bootstrap confidence intervals
        collapsed_samples = np.where(gap_ratio < gap_threshold)[0]
        fraction_mem = len(collapsed_samples) / len(gap_ratio)
        
        if len(collapsed_samples) > 0:
            fraction_mem, std_frac, lower, upper = bootstrap_mean_se(
                gap_ratio.numpy(), gap_threshold
            )
        else:
            std_frac = 0.0
            lower = 0.0
            upper = 0.0

        pbar.set_description(f'Fmem = {fraction_mem*100:.2f}% ± {std_frac*100:.2f}')

        # Write results to file
        with open(file_fc, "a") as myfile:
            myfile.write(f"\n{tau:d}\t{fraction_mem*100:.3f}\t{std_frac*100:.5f}\t"
                        f"{lower*100:.5f}\t{upper*100:.5f}")


def main():
    """Main function to compute fraction collapsed."""
    # Parse arguments
    args = parse_arguments()
    print("Arguments:", args)
    
    # Load configuration
    config = cfg.load_config(args.dataset)
    config.IMG_SHAPE = (1, args.img_size, args.img_size)
    config.n_images = args.num
    config.BATCH_SIZE = min(args.batch_size, config.n_images)
    config.OPTIM = args.optim
    config.LR = args.learning_rate
    config.DEVICE = args.device
    
    # Model type string for paths
    type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/'.format(
        config.DATASET, args.img_size, config.n_images, args.nbase, 
        config.OPTIM, config.BATCH_SIZE, config.LR, args.index
    )
    
    # Create output directory and file
    path_file = config.path_save + type_model + 'Memorization/'
    file_fc = path_file + 'fraction_memorized.txt'
    if os.path.exists(file_fc):     # Remove existing file
        os.remove(file_fc)
    os.makedirs(path_file, exist_ok=True)
    
    # Define training times to analyze
    training_times = cfg.get_training_times()
    
    print(f"Computing memorization fraction for {len(training_times)} checkpoints...")
    print(f"Model: {type_model}")
    print(f"Output file: {file_fc}")
    
    # Load training data
    train_images, _ = cfg.load_training_data(config, args.index)
    train_images = train_images[:config.n_images, :, :, :].to(config.DEVICE)
    
    # Setup diffusion configuration
    df = dm.DiffusionConfig(
        n_steps=config.TIMESTEPS,
        img_shape=config.IMG_SHAPE,
        device=config.DEVICE,
    )
    
    # Compute fraction collapsed for each checkpoint
    compute_fraction_mem(
        training_times=training_times,
        train_images=train_images,
        type_model=type_model,
        config=config,
        file_fc=file_fc,
        nsamples=args.Nsamples,
        sample_size=args.batch_sample_size,
        gap_threshold=args.gap_threshold
    )
    
    print("Memorization fraction computation completed!")


if __name__ == "__main__":
    main()