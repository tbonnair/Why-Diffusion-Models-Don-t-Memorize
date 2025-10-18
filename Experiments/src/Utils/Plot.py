import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import torch
import torchvision

plt.rcParams['figure.figsize'] = [6,6]
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight']= 'normal'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits']=(-6, 6)
mpl.rcParams['axes.formatter.use_mathtext']=True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

def imshow(images, mean=.5, std=.5):
    img = torchvision.utils.make_grid(images)
    # Unnormalize the image
    if images.shape[1] > 1:            # Multi channels
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
    else:
        img = img * std[0] + mean[0]      # Single channel
    
    # Plot it
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    ax.set_axis_off()
    return fig

def cvtImg(img):
    # Unnormalize the image 
    img = img.permute([0, 2, 3, 1])
    img = img - img.min()
    img = (img / img.max())
    
    # Return it as a numpy array
    return img.numpy().astype(np.float32)


def show_examples(x):
    fig, ax = plt.subplots(figsize=(10, 10))
    imgs = cvtImg(x) # Unnormalize images
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
    return fig


# Compute nearest neighbors of an image in a dataset 
def compute_knn(dataset, x, k=3):
    dist = torch.norm(dataset - x, dim=(1, 2, 3), p=None)
    knn = dist.topk(k, largest=False)
    return knn
