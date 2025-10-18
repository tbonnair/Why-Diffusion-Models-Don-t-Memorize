import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import Plot

# ====================================================================
# Training Configuration Class
# ====================================================================
class TrainingConfig:
    '''
    TrainingConfig: Class containing all information on the data, device, LR,
    Number of SGD steps, paths for saving, etc.
    '''
    DATASET = ''                # Dataset name (MNIST, CIFAR, Imagenet, CelebA)
    IMG_SHAPE = (3, 32, 32)     # Fixed input image size
    BATCH_SIZE = 128            # Batch size
    DEVICE = 'cuda:0'           # Name of the device to be used
    LR = 1e-4                   # Learning rate
    N_STEPS = int(1e5)+1        # Number of SGD steps
    TIMESTEPS = 1000            # Define number of diffusion timesteps
    path_save = ''              # Path for saving plots and models
    path_data = ''              # Path for the data
    CENTER = True               # Whether the dataset should be centered
    STANDARDIZE = False         # Whether the dataset should be standardized
    n_images = 500              # Number of images per class
    NUM_WORKERS = 2             # Number of workers
    
    mean = 0                    # Mean of the dataset (to be computed)
    std = 0                     # Std of the dataset (to be computed)

def get(element: torch.Tensor, t: torch.Tensor, dim: int=2):
    '''
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    '''
    ele = element.gather(-1, t)
    if dim == 4:
        return ele.reshape(-1, 1, 1, 1)
    elif dim == 3:
        return  ele.reshape(-1, 1, 1)
    elif dim == 2:
        return  ele.reshape(-1, 1)


# ====================================================================
# Diffusion class
# ====================================================================
class DiffusionConfig:
    '''
    ClassDiffusion: Class containing information related to the 
    diffusion process (number of steps, device, variance, etc.)
    '''
    def __init__(self, n_steps=1000, img_shape=(3, 32, 32), device='cpu'):
        self.n_steps = n_steps
        self.img_shape = img_shape
        self.device = device
        self.initialize()

    def initialize(self):
        self.beta  = self.linear_schedule()  # Linear or fixed
        self.alpha = 1 - self.beta

        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

        self.times = 1 - np.linspace(0, 1.0, self.n_steps + 1)

    def linear_schedule(self, b0=1e-4, bT=2e-2):
        '''
        Linear schedule from b0 to bT as used in Ho et al., 2020
        '''
        scale = 1000 / self.n_steps
        beta_start = scale * b0
        beta_end = scale * bT
        return torch.linspace(beta_start, beta_end, self.n_steps, 
                              dtype=torch.float32,
                              device=self.device)
    
    def fixed_schedule(self, b=6e-3):
        return torch.linspace(b, b, self.n_steps, 
                              dtype=torch.float32, device=self.device)

    def betas_for_alpha_bar(self, alpha_bar, max_beta=0.999):
        '''
        Constructs a beta schedule from the definition of alpha_bar
        '''
        betas = []
        for i in range(self.n_steps):
            t1 = i / self.n_steps
            t2 = (i + 1) / self.n_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

# ====================================================================
# Diffusion functions
# ==================================================================== 
def forward_diffusion(df, x0, timesteps, config):
    dim = len(x0.shape)
    # Generate noise realisation with the same size as a batch of images
    eps = torch.randn_like(x0)
    
    # Apply the forward diffusion kernel at times timesteps
    mean    = get(df.sqrt_alpha_cumulative, timesteps, dim) * x0  # Image scaled by exp(-t)
    std_dev = get(df.sqrt_one_minus_alpha_cumulative, timesteps, dim) # Noise scaled by sqrt(1-exp(-2t))
    sample_a  = mean + std_dev * eps    # step t of the forward process
    
    # Return the noisy image and the noise realisation
    return sample_a, eps


@torch.no_grad()
def sample_diffusion(model, n_images=25, config=TrainingConfig()):
    # Gaussian random fields
    x_init = torch.randn(n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], config.IMG_SHAPE[2]).to(config.DEVICE)
    x = x_init.clone()
    for i in trange(config.TIMESTEPS):
        t = i
        # For each time step, generate a denoised image
        x = model(x, torch.full((n_images, 1), t, dtype=torch.long, device=config.DEVICE))
    return x, x_init


@torch.no_grad()
def sample_diffusion_from_noise(model, n_images=25, config=TrainingConfig(), 
                                df=DiffusionConfig(), dim=3):
    
    # Generate n_images starting points from N(0, 1)
    if dim == 4: # Assumes [B, C, H, W] for 2d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1], 
                             config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3: # Assumes [B, C, N] for 1d
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], 
                             config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2: # Assumes [B, N] for 1d (no channels)
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)
    x = x_init.clone()
    
    model.eval()
    for t in reversed(range(0, config.TIMESTEPS)):
        # Time tensor
        ts = torch.ones(n_images, dtype=torch.long, device=config.DEVICE) * t
        
        # Generate one realisation of the noise
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        
        # Predict the noise at times ts
        eps_ts = model(x, ts)
        
        # Get scaling quantities
        beta_t                            = get(df.beta, ts, dim)
        one_by_sqrt_alpha_t               = get(df.one_by_sqrt_alpha, ts, dim)
        sqrt_one_minus_alpha_cumulative_t = get(df.sqrt_one_minus_alpha_cumulative, ts, dim) 
        
        # Langevin sampling from Ho et al., 2020
        x = (one_by_sqrt_alpha_t 
             * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) 
                * eps_ts) + torch.sqrt(beta_t) * z)
    return x, x_init


@torch.no_grad()
def sample_diffusion_from_noise_DDIM(model, n_images=25, config=TrainingConfig(), 
                                df=DiffusionConfig(), dim=3, eta=0.0, ddim_steps=None):
    """
    Generates images using the DDIM sampling procedure with a subsampled schedule.
    
    Parameters:
      model: The noise prediction network.
      n_images: Number of images to generate.
      config: TrainingConfig with attributes such as IMG_SHAPE, TIMESTEPS, DEVICE, etc.
      df: DiffusionConfig with diffusion schedule tensors
      dim: Dimensionality of the tensor (e.g., 2 for [B, N], 3 for [B, C, N], 4 for [B, C, H, W]).
      eta: Hyperparameter controlling stochasticity (eta=0 yields a deterministic process).
      ddim_steps: Number of steps S to use for sampling. If None, use all timesteps.
      
    Returns:
      x: The final generated images.
      x_init: The initial noise samples.
    """
    # Determine the number of steps to use
    total_steps = config.TIMESTEPS
    if ddim_steps is None:
        ddim_steps = total_steps

    # Create a schedule of timesteps (linearly spaced and then reversed)
    time_steps = np.linspace(0, total_steps - 1, ddim_steps, dtype=int)[::-1]
    time_steps = list(time_steps)

    # Generate initial noise
    if dim == 4:  # Assumes [B, C, H, W] for 2D images
        x_init = torch.randn(n_images, config.IMG_SHAPE[0], config.IMG_SHAPE[1],
                             config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3:  # Assumes [B, C, N] for 1D signals with channels
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                             config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2:  # Assumes [B, N] for 1D signals (no channels)
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)
    x = x_init.clone()

    model.eval()
    # Iterate over the subsampled schedule of timesteps.
    for i in range(len(time_steps) - 1):
        t = time_steps[i]
        t_prev = time_steps[i + 1]

        # Create time tensors for current and previous timesteps
        ts = torch.full((n_images,), t, dtype=torch.long, device=config.DEVICE)
        ts_prev = torch.full((n_images,), t_prev, dtype=torch.long, device=config.DEVICE)

        # Predict the noise at time t
        eps_ts = model(x, ts)

        # Get scaling factors for current timestep t:
        sqrt_one_minus_alpha_cum_t = get(df.sqrt_one_minus_alpha_cumulative, ts, dim)
        sqrt_alpha_cum_t = get(df.sqrt_alpha_cumulative, ts, dim)

        # Estimate the clean image (x0) at time t:
        x0_pred = (x - sqrt_one_minus_alpha_cum_t * eps_ts) / sqrt_alpha_cum_t

        # Get scaling factors for previous timestep t_prev:
        sqrt_one_minus_alpha_cum_t_prev = get(df.sqrt_one_minus_alpha_cumulative, ts_prev, dim)
        sqrt_alpha_cum_t_prev = get(df.sqrt_alpha_cumulative, ts_prev, dim)

        # Compute sigma_t as per DDIM:
        sigma_t = eta * torch.sqrt(
            (sqrt_one_minus_alpha_cum_t_prev ** 2 / (sqrt_one_minus_alpha_cum_t ** 2))
            * (1 - (sqrt_alpha_cum_t ** 2) / (sqrt_alpha_cum_t_prev ** 2))
        )

        # Add noise only if t is not the final step
        z = torch.randn_like(x) if t_prev > 0 else torch.zeros_like(x)

        # DDIM update (Eq. 12 from arxiv:2010.02502)
        # x_{t_prev} = sqrt(ᾱ_{t_prev}) * x0_pred +
        #              sqrt(1 - ᾱ_{t_prev} - σ_t^2) * eps_ts +
        #              σ_t * z
        x = (sqrt_alpha_cum_t_prev * x0_pred +
             torch.sqrt(torch.clamp(sqrt_one_minus_alpha_cum_t_prev ** 2 - sigma_t ** 2, min=1e-8)) * eps_ts +
             sigma_t * z)

    return x, x_init

 
#==========================================
# Training functions
#==========================================
def train_one_batch(X, model, optimizer, loss_fn, 
                    config=TrainingConfig(), 
                    df=DiffusionConfig()):
    model.train()
    
    # Generate random times
    if config.mode == 'normal':
        ts = torch.randint(low=1, high=config.TIMESTEPS, size=(X.shape[0],), device=config.DEVICE)
    elif config.mode == 'fixed_time':
        # If mode is fixed_time, only train for this time_step
        ts = torch.ones((X.shape[0],), dtype=torch.long, device=config.DEVICE) * config.time_step
        
    # Extract noisy images from times t
    X_t, noise_t = forward_diffusion(df, X, ts, config)
    X_t = X_t.to(config.DEVICE)
    
    # Apply the model
    Y = model(X_t.float(), ts)
    
    # The loss is comparing the predicted and true noises
    loss = loss_fn(noise_t, Y)
    
    # Update parameters of the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Return the current loss and the batch of noisy images
    return loss.detach().item(), X_t


def train(model, trainloader, optimizer, config, df, loss_fn,
          sweep=1., times_save=[], offset=0, suffix='', generate=False):
    
    n_steps = offset    # Number of SGD steps
    k_steps = 100       # Number of steps before printing
    
    bar = trange(config.N_STEPS, leave=True, position=0)
    bar.update(offset)
    
    while n_steps < config.N_STEPS:
        for i, X in enumerate(trainloader):
            X = X.to(config.DEVICE)
            loss, _, = train_one_batch(X, model, optimizer, loss_fn, config, df)
            n_steps += 1            # Update number of steps
            
            shallSave = n_steps in times_save #(n_steps%save_every == 0)
            if n_steps >= config.N_STEPS:
                shallSave = 1
            
            if shallSave == 1:
                # Save model
                p = config.path_save + 'Models/' + suffix + 'Model_{:d}'.format(n_steps)
                torch.save(model.state_dict(), p)
                
                if generate:
                    # Sample a small batch and save it to check quality visually
                    if len(X.shape) == 4: # For images, assumes [B, C, H, W]
                        samples, samples_init = sample_diffusion_from_noise(model, 16, config, df, dim=4)
                        fig = Plot.imshow(samples.cpu(), config.mean, config.std)
                        fig.savefig(config.path_save + 'Images/' + suffix + 'Sample_{:d}.pdf'.format(n_steps), bbox_inches='tight')
                        plt.close('all')
                        
            # Update the bar (every k steps)
            if n_steps%k_steps == 0:
                bar.set_description(f'loss: {loss:.5f}, n_steps: {n_steps:d}')
                bar.update(k_steps)
            
            # If we performed all the steps, exit
            if n_steps >= config.N_STEPS:
                break
            
    # Return nothing
    return


def train_one_batch_coupling(X, X2, model, model2, optimizer, optimizer2, loss_fn, 
                    config=TrainingConfig(), df=DiffusionConfig(), tau=0):
    model.train()
    model2.train()
    
    # Generate random times
    ts = torch.randint(low=1, high=config.TIMESTEPS, size=(X.shape[0],), device=config.DEVICE)
  
    # Noise data at time t for X
    X_t, noise_t = forward_diffusion(df, X, ts, config)
    X_t = X_t.to(config.DEVICE)
    Y = model(X_t.float(), ts)
    Y2 = model2(X_t.float(), ts)
    loss = loss_fn(noise_t, Y, Y2.detach(), tau)
    
    # Update parameters of the first model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Second model update
    X_t2, noise_t2 = forward_diffusion(df, X2, ts, config)
    X_t2 = X_t2.to(config.DEVICE)
    optimizer.zero_grad()
    optimizer2.zero_grad()
    Y2 = model2(X_t2, ts)
    Y = model(X_t2, ts)
    loss2 = loss_fn(noise_t2, Y2, Y.detach(), tau)
    loss2.backward()                 # Backward pass
    optimizer2.step()                # Update the weights
    
    # Return the current loss and the batch of noisy images
    return loss.detach().item(), X_t


def compute_empirical_score(samples, data_samples, t, df, device='cpu'):
    '''
    Parameters
    ----------
    samples : [B, C, H, W] pyTorch tensor
        Contains the B samples where to evaluate the score.
        
    data_samples : [n, C, H, W] pyTorch tensor
        Contains the n training samples of size CxHxW.
        
    t : int
        Timestep in [0, T].
        
    df : DiffusionConfig object
        Contains all the configuration information about the Diffusion.
        
    device : string
        Which device to be used (cpu or cuda).

    Returns
    -------
    Femp : [B, C, H, W] pyTorch tensor
        Contains the empirical score estimated at points 'samples'
        using training data 'data_samples'.

    '''
    n = len(data_samples)
    d = int(np.prod(data_samples.shape[1::]))
    n_gen = len(samples)
    dim = len(data_samples.shape)
    Dt = 1 - df.alpha_cumulative.to('cpu').numpy()
    
    ts = torch.ones(size=(n,), device=device,
                    dtype=torch.long) * t
    Xs = get(df.sqrt_alpha_cumulative.to(device), t=ts.to(device), dim=dim) * data_samples
    Xs = Xs.reshape(-1, d).to(device)
    s = samples.reshape(-1, 1, d).to(device)
    
    # Tensorized version (12/09/24)
    dist = torch.norm(s - Xs, dim=2, p=2)
    
    dst_sq = dist.cpu()**2
    mindist = dst_sq.min(1).values.reshape(-1, 1)
    sumexp = np.exp(-(dst_sq-mindist)/2/Dt[t]).sum(1).reshape(-1, 1)
    log_Pt = -mindist/2/Dt[t] + np.log(sumexp) - np.log(n) - d/2*np.log(2*np.pi*Dt[t])
    
    # Compute derivative of the probability
    diff = s - Xs
    
    weighted_sumexp = (diff.cpu()*np.exp(-(dst_sq-mindist)/2/Dt[t]).reshape(n_gen, n, -1)).sum(1)
    signs = (torch.sign(weighted_sumexp) + 1) / 2 # To have 0 and 1
    log_sum = np.log(torch.abs(-weighted_sumexp)) + 1j*np.pi*signs
    # log_Ptprime = - np.log( n * Dt[t] * (2*np.pi*Dt[t])**(d/2) ) - mindist/2/Dt[t] + log_sum
    log_Ptprime = - np.log(n * Dt[t]) - (d/2)*np.log(2*np.pi*Dt[t]) - mindist/2/Dt[t] + log_sum
    
    Femp = np.exp(log_Ptprime - log_Pt).real
    Femp = Femp.reshape(samples.shape)
    
    return Femp
    
