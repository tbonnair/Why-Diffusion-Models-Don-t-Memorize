#%%
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import os
import numpy as np
import argparse

sys.path.insert(1, '../Utils/')      # In case we run from Experiments/
import Unet
import Plot
import Diffusion
import loader
import cfg
from numpy.random import default_rng
from torch.utils.data import Dataset, DataLoader
import GMM_data_scores as GMM
import TinyModels as TM

#%%

parser = argparse.ArgumentParser("Diffusion on Gaussian Mixture Model with simple time model.")
parser.add_argument("-n", "--num", help="Number of training data to generate (should be pair)", type=int)
parser.add_argument("-d", "--dim", help="Dimension of the training data", type=int)
parser.add_argument("-s", "--seed", help="Seed for the dataset", type=int)
parser.add_argument("-de", "--d_embed", help="Width of the neural network", type=int)
parser.add_argument("-O", "--optim", help="Optimisation type", type=str) # Adam or SGD_Momentum
parser.add_argument("-B", "--bs", help="Batch size", type=int)
parser.add_argument("-t", "--time", help="Diffusion timestep", type=int)
args = vars(parser.parse_args())
print(args)

# Get parameters
n = args['num']
d = args['dim']
seed = args['seed']
n_base = args['d_embed']
optim = args['optim']
device = 'cuda:0'
BATCH_SIZE = args['bs']
time_step = args['time']
if time_step == -1:
    mode = 'normal'
else:
    mode = 'fixed_time'

# Overwrite config file
config = Diffusion.TrainingConfig()
DATASET = 'GMM'
config.DATASET = DATASET
config.n_images = n
config.IMG_SHAPE = (1, d)
config.BATCH_SIZE = BATCH_SIZE
config.N_STEPS = int(4e6)
config.LOSS_SCORE_EMP = False
config.OPTIM = optim
if config.OPTIM == 'SGD_Momentum':
    config.LR = 6e-3
else:
    config.LR = 6e-4
config.mode = mode
config.time_step = time_step

suffix = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_B{:d}_t{:d}/'.format(config.DATASET, d, config.n_images, n_base,
                                       config.OPTIM, seed, config.LR, BATCH_SIZE, time_step)
config.DEVICE = device
config.path_save = '../../Saves/'

# Create path to images and model save
path_models = config.path_save + '/Models/' + suffix
os.makedirs(path_models, exist_ok=True)

os.system('cp run_GMM.py {:s}'.format(path_models + '_run_GMM.py'))
os.system('cp ../Utils/loader.py {:s}'.format(path_models + '_loader.py'))
os.system('cp ../Utils/cfg.py {:s}'.format(path_models + '_cfg.py'))

# Load data
mu = 1
sigma = 1
X_train, y_train = GMM.generate_GMM(n, d, mu, sigma, seed)
trainset = X_train.to(torch.float32).to(device)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# In[] Model definition
model = TM.SimpleTimeModel(d=d, d_model=n_base).to(device)
n_params = sum(p.numel() for p in model.parameters())
print('Total number of parameters = {:.2f}K'.format(n_params/1e3))

# In[] Training and saving

if __name__ == '__main__':
    if config.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    elif config.OPTIM == 'SGD_Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.95)
        
    df = Diffusion.DiffusionConfig(
        n_steps                 = config.TIMESTEPS,
        img_shape               = config.IMG_SHAPE,
        device                  = config.DEVICE,
    )
    loss_fn = nn.MSELoss()
    
    sweeping = 1.0
    times_save1 = np.arange(0, 5000, 250).astype(int)
    times_save2 = np.arange(5000, config.N_STEPS, 5000).astype(int)
    times_save = np.hstack((times_save1, times_save2))
    
    offset = 0
    Diffusion.train(model, train_loader, optimizer, config, df, 
                    loss_fn, sweeping, times_save, offset, suffix)