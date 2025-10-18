import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
from tqdm import tqdm

sys.path.insert(1, '../Utils/')      # In case we run from Experiments/Generation
import Diffusion
import Unet
import cfg
import argparse
import loader
import calc

# Parse arguments (Ngen, device, DATASET, N, n_base, save_every)
parser = argparse.ArgumentParser()
parser.add_argument('--n_base', type=int,
                    help='Base number of filters in the U-Net.')
parser.add_argument('--device', type=str,
                    help='Device used to load and apply the model.')

parser.add_argument('--N', type=int,
                    help='Number of images used to train the model.')
parser.add_argument('--O', type=str,
                    help='Optimization mode used to train the model (Adam or SGD_Momentum)')
parser.add_argument('--LR', type=float,
                    help='Learning rate used to train the model')
parser.add_argument('--B', type=int,
                    help='Batch size used to train the model')
parser.add_argument('--d', type=int,
                    help='Dimension of the input data')
parser.add_argument('--index', type=int,
                    help='Index of the trained model (0 or 1)')
parser.add_argument('--dataset', type=str,
                    help='Dataset used to train the model.')
parser.add_argument('--Nsamples', type=int,
                    help='Number of samples to generate.')

args = parser.parse_args()
print(args)
DATASET = args.dataset
config = cfg.load_config(DATASET)   # Load base config for this dataset
n_base = int(args.n_base)
config.DEVICE = args.device
config.n_images = int(args.N)
Nsamples = int(args.Nsamples)
size = int(args.d)
config.OPTIM = args.O
config.BATCH_SIZE = int(args.B)
config.LR = float(args.LR)
index = int(args.index)

if not Nsamples % 100 == 0:
    raise TypeError('Nsamples should be a multiple of 100.')

# Load diffusion config for these data
df = Diffusion.DiffusionConfig(
    n_steps                 = config.TIMESTEPS,
    img_shape               = config.IMG_SHAPE,
    device                  = config.DEVICE,
)

# Load model on the device
type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/'.format(config.DATASET, size,
                                     config.n_images, n_base, config.OPTIM, config.BATCH_SIZE,
                                     config.LR, index)

model_diffusion = Unet.UNet(
    input_channels          = config.IMG_SHAPE[0],
    output_channels         = config.IMG_SHAPE[0],
    base_channels           = n_base,
    base_channels_multiples = (1, 2, 4),
    apply_attention         = (False, True, True),
    dropout_rate            = 0.1,
)
model_diffusion.to(config.DEVICE)

print('Generating {:d} samples'.format(Nsamples))

# Generate samples
batch_gen = 100
Ns = Nsamples // batch_gen

# Define the training times to sample models
# TODO: make it better than this hardcoded stuff
a = np.logspace(np.log10(250+1), 4, 10)
training_times1 =  calc.unique_modulus(a, 250).astype(int)
a = np.logspace(4, 6, 90)
training_times2 =  calc.unique_modulus(a, 5000).astype(int)
a = np.logspace(6, 7, 20)
training_times3 =  calc.unique_modulus(a, 5000).astype(int)
training_times = np.hstack((training_times1, training_times2, training_times3))
training_times = np.unique(training_times)[::2]

# Loop over training times
for (j, checkpoint_id) in enumerate(training_times):
    print(r'Training time = {:d} ({:d}/{:d})'.format(checkpoint_id, j, len(training_times)))
    
    # Load the model
    try:
        model_suffix = type_model + '/Model_{:d}'.format(checkpoint_id)
        path_model_diffusion = config.path_save + '/Models/' + model_suffix
        model_diffusion = loader.load_model(model_diffusion, path_model_diffusion)
    except:
        raise NameError('The checkpoint does not exist: {:s}'.format(path_model_diffusion))
    
    # Loop for generation at the current checkpoint
    for i in range(0, Ns):
        path_save = config.path_save + '/Samples/' + type_model +'/{:d}/'.format(checkpoint_id)
        doesExist = os.path.exists(path_save)
        if not doesExist:
            os.makedirs(path_save)
        
        print('Sample {:d}/{:d}'.format(i, Ns))
        samples_gen, samples_init = Diffusion.sample_diffusion_from_noise_DDIM(model_diffusion,
                                            n_images=batch_gen,
                                            config=config,
                                            df=df,
                                            dim=4,
                                            eta=0.0,            # Deterministic trajectories
                                            ddim_steps=100)     # Number of steps reduced (much faster)
        # Save initial samples
        path = path_save + str(config.TIMESTEPS)
        # Create dir if does not exist
        doesExist = os.path.exists(path)
        if not doesExist:
            os.makedirs(path)
        torch.save(samples_init, path + '/samples_a_{:d}'.format(i))
        
        # Save the generated image
        path = path_save + 'generated'
        # Create dir if does not exist
        doesExist = os.path.exists(path)
        if not doesExist:
            os.makedirs(path)
        torch.save(samples_gen, path + '/samples_a_{:d}'.format(i))

print('Done!')
