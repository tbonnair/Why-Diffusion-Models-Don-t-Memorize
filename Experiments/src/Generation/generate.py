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

# Parse arguments (Ngen, device, DATASET, N, n_base, save_every)
parser = argparse.ArgumentParser("Generation of samples from trained diffusion models.")

parser.add_argument("-n", "--num", help="Number of training data", type=int)
parser.add_argument("-i", "--index", help="Index for the dataset (0 or 1)", type=int)
parser.add_argument("-s", "--img_size", help="Size of the images used to train", type=int)
parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float)
parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str)
parser.add_argument("-W", "--nbase", help="Number of base filters", type=str)
parser.add_argument("-t", "--time", help="Diffusion timestep", type=int)
parser.add_argument("-B", "--batch_size", type=int,
                    help="Batch size used to train the model")
parser.add_argument('-D', '--dataset', type=str,
                    help='Dataset used to train the model.')
parser.add_argument('-Ns', '--Nsamples', type=int,
                    help='Number of samples to generate (should be multiple of 100).')
parser.add_argument('--device', type=str,
                    help='Device used to load and apply the model.', default='cuda:0')

args = parser.parse_args()
print(args)
DATASET = args.dataset
config = cfg.load_config(DATASET)   # Load base config for this dataset
n_base = int(args.nbase)
config.DEVICE = args.device
config.n_images = int(args.num)
Nsamples = int(args.Nsamples)
size = int(args.img_size)
config.OPTIM = args.optim
config.BATCH_SIZE = int(args.batch_size)
config.LR = float(args.learning_rate)
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
training_times = cfg.get_training_times()

# Loop over training times
for (j, checkpoint_id) in enumerate(training_times):
    print(r'Training time = {:d} ({:d}/{:d})'.format(checkpoint_id, j, len(training_times)))
    
    # Load the model
    try:
        model_suffix = '/Model_{:d}'.format(checkpoint_id)
        path_model_diffusion = config.path_save + type_model + '/Models/' + model_suffix
        model_diffusion = loader.load_model(model_diffusion, path_model_diffusion)
    except:
        raise NameError('The checkpoint does not exist: {:s}'.format(path_model_diffusion))
    
    # Loop for generation at the current checkpoint
    for i in range(0, Ns):
        path_save = config.path_save + type_model + '/Samples/' + '{:d}/'.format(checkpoint_id)
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
