import Diffusion

def load_config(DATASET):
    config = Diffusion.TrainingConfig()
    config.DATASET = DATASET             # Dataset name
    
    # Fields common to all datasets
    config.DEVICE = 'cuda:0'
    config.LR = 1e-4
    config.N_STEPS = int(1e6)
    config.path_save = '../Saves/'
    
    if DATASET == 'CelebA':
        config.IMG_SHAPE = (1, 32, 32)
        config.BATCH_SIZE = 256
        config.path_data = '../../../data/celeba/' #img_align_celeba'
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 50000
        
    else:
        raise Exception('Dataset {:s} not implemented'.format(DATASET))
        
    return config
