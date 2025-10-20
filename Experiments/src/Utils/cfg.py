import Diffusion

def load_config(DATASET):
    config = Diffusion.TrainingConfig()
    config.DATASET = DATASET             # Dataset name
    
    if DATASET == 'CelebA':
        config.path_save = '../../Saves/'
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
