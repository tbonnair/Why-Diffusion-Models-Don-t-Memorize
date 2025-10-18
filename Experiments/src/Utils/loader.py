import numpy as np
import torch
import torchvision.transforms as transforms

# For CelebA
import os
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted


# ===================================================================
#   Helper functions for loading datasets
# ===================================================================

# Create a custom Dataset class
class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Get image names
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)
    self.attr = np.loadtxt(root_dir+'/../list_attr_celeba.txt', skiprows=2,
                           usecols=np.arange(1, 41))

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)
      
    # label = self.attr[idx, 20]  # Male or fem
    # if label == -1:
    #     label = 0

    return img#, label



# ===================================================================
#   Loading the datasets
# ===================================================================
def load_CelebA(config, loadtest=False, ntest=2048, index=0):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    loadtest : TYPE, optional
        Whether or not to load a test set. The default is False.
    ntest : int, optional
        Number of test images to load. The default is 2048.
    index : int, optional
        Index of the subset to load. The default is 0.

    Returns
    -------
    trainset : torchvision.datasets
        Subset of the training set containing config.n_images for each class.
    testset : torchvision.datasets
        Subset of the training set containing test images for each class.
    '''
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(config.IMG_SHAPE[1]),
         transforms.CenterCrop(config.IMG_SHAPE[1]),
         transforms.Grayscale(1),
         ])
    
    celeba_dataset = CelebADataset(config.path_data, transform)
    
    indices = np.arange(index*config.n_images, (index+1)*config.n_images) # Load images between index and index+1 times the number of data
    trainset = torch.utils.data.Subset(celeba_dataset, indices)
    testset = None
    
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([1.0, 1.0, 1.0])
    if config.CENTER:
        tmploader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                                  shuffle=False, num_workers=1)
        t_data = next(iter(tmploader))
        
        mean = torch.mean(t_data, axis=[0, 2, 3])
        if config.STANDARDIZE:
            std = torch.std(t_data, axis=[0, 2, 3])
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(config.IMG_SHAPE[1]),
             transforms.CenterCrop(config.IMG_SHAPE[1]),
             transforms.Normalize(mean, std),
             transforms.Grayscale(1),
             ])
        
        # Reload data
        celeba_dataset = CelebADataset(config.path_data, transform)
        trainset = torch.utils.data.Subset(celeba_dataset, indices)
        
        if loadtest:
            # indices_test = np.arange(config.n_images, config.n_images + ntest)
            indices_test = np.arange(-ntest, 0)
            testset = torch.utils.data.Subset(celeba_dataset, indices_test)
        
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return trainset, testset


class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x = self.base[idx]
        if self.transform:
            x = self.transform(x)
        return x

def load_CelebA_pt(config, full_tensor, loadtest=False, ntest=2048, index=0):
    '''
    Parameters
    ----------
    config : class Diffusion.TrainingConfig()
        Contains all the training information
    full_tensor : torch.Tensor
        Tensor containing the full dataset.
    loadtest : TYPE, optional
        Whether or not to load a test set. The default is False.
    ntest : int, optional
        Number of test images to load. The default is 2048.
    index : int, optional
        Index of the subset to load. The default is 0.

    Returns
    -------
    trainset : Transformed tensor of 
        Subset of the training set containing config.n_images images.
    testset : torchvision.datasets
        Subset of the training set containing ntest test images.
    '''
    
    # Load training and test sets
    train_images = full_tensor[index*config.n_images:(index+1)*config.n_images]
    test_images = None
    if loadtest:
        test_images = full_tensor[-ntest:]
    
    # Center and standardize the data
    mean = torch.zeros(config.IMG_SHAPE[0])
    std = torch.ones(config.IMG_SHAPE[0])
    if config.CENTER:
        mean = torch.mean(train_images, axis=[0, 2, 3])
        if config.STANDARDIZE:
            std = torch.std(train_images, axis=[0, 2, 3])
        
        transform = transforms.Compose(
            [transforms.Normalize(mean, std),])
        
        # Transform the trainset and testset
        train = TransformedDataset(train_images, transform=transform)
        test = None
        if loadtest:
            test = TransformedDataset(test_images, transform=transform)
        
    # Store mean and std
    config.mean = mean
    config.std = std
    
    return train, test
    



# ===================================================================
#   Loading the model
# ===================================================================
def load_model(model: torch.nn.Module, path_checkpoint: str, verbose: bool = True):
    state_dict = torch.load(path_checkpoint, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # Remove 'module.'
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict)
    if verbose:
        print('Loading initial state at {:s}'.format(path_checkpoint))
    return model