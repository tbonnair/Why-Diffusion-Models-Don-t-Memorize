import Diffusion
import calc
import numpy as np
import torch
import loader

def load_config(DATASET):
    config = Diffusion.TrainingConfig()
    config.DATASET = DATASET             # Dataset name
    
    if DATASET == 'CelebA':
        config.path_save = '../../Saves/'          # Path to save results from Experiments/src/FOLDER/
        config.IMG_SHAPE = (1, 32, 32)
        config.BATCH_SIZE = 512
        config.path_data = '../../Data/CelebA/'    # Path to CelebA dataset from Experiments/src/FOLDER/
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 1024
        config.BATCH_SIZE = min(512, config.n_images)
        config.N_STEPS = int(2e6)
        config.LOSS_SCORE_EMP = False
        config.OPTIM = 'SGD_Momentum'
        config.LR = 1e-2
        config.mode = 'normal'
        config.time_step = -1
        config.DEVICE = 'cuda:0'
        config.TIMESTEPS = 1000
        
    else:
        raise Exception('Dataset {:s} not implemented'.format(DATASET))
    return config

def get_training_times():
    """Generate training time checkpoints to save the models (used to generate and compute metrics as well)."""
    a = np.logspace(np.log10(250+1), 4, 10)
    training_times1 = calc.unique_modulus(a, 250).astype(int)
    a = np.logspace(4, 6, 90)
    training_times2 = calc.unique_modulus(a, 5000).astype(int)
    a = np.logspace(6, 7, 10)
    training_times3 = calc.unique_modulus(a, 5000).astype(int)
    training_times = np.hstack((0, training_times1, training_times2, training_times3))
    return np.unique(training_times)#[::2]

def load_training_data(config, index, loadtest=False):
    """Load and prepare training data."""
    # loading_func = 'loader.load_{:s}(config, index={:d})'.format(config.DATASET, index)
    # trainset, _ = eval(loading_func)
        
    # Torch Tensor version
    size = config.IMG_SHAPE[1]
    all_images = torch.load(config.path_data + '{:s}{:d}.pt'.format(config.DATASET, size))
    trainset, testset = loader.load_CelebA_pt(config, all_images, loadtest=loadtest, index=index)
    
    return trainset, testset