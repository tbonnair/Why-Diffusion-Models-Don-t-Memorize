import torch
from torch import nn
import numpy as np

import sys
sys.path.insert(1, './Utils/')
import Diffusion
# --------------------------- 
#            Data
# ---------------------------
# Code for generating data
def generate_GMM(n, d, mu=1, sigma=1, seed=0):
    """
    Generate a two-component Gaussian Mixture Model dataset.
    
    Args:
        n (int): Total number of samples (must be even).
        d (int): Dimensionality of the data.
        mu (float): Mean offset (controls separation of clusters).
        sigma (float): Standard deviation of each Gaussian component.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        X (torch.Tensor): Dataset of shape (n, d).
        labels (torch.Tensor): Cluster labels (0 or 1), shape (n,).
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if n % 2 != 0:
        raise ValueError("The number of samples n should be even.")
        
    n_per_cluster = int(n // 2)
    
    # Define means
    mu1 = torch.ones(d) * mu            # mu of order one
    mu2 = -mu1                          # Opposite mean
    
    # Sample points
    X1 = torch.randn(n_per_cluster, d) * sigma + mu1
    X2 = torch.randn(n_per_cluster, d) * sigma + mu2

    # Stack the dataset and create labels
    X = torch.vstack((X1, X2))
    labels = torch.cat((torch.zeros(n_per_cluster), torch.ones(n_per_cluster))).long()
    return X, labels


# --------------------------- 
#           Scores
# ---------------------------
# Generalization score
def compute_generalization_score_GMM(x, mu, t, device='cpu'):
    '''
    Compute the population score of a 2-Gaussian Mixture Model
    Assuming P0 centred on +mu and -mu, and unit variance.
    '''
    Gamma_t = 1 # 1 - np.exp(-2*t)
    d = x.shape[1]
    mean = (torch.ones(d, device=device)*mu*np.sqrt(d))
    m = x @ (mean * np.exp(-t))
    mean = mean.expand(x.shape)
    return (mean*np.exp(-t) * torch.tanh(m / Gamma_t).reshape(-1, 1) - x) / Gamma_t

def compute_empirical_score(samples, data_samples, t, device='cpu'):
    '''
    Parameters
    ----------
    samples : [B, C, H, W] pyTorch tensor
        Contains the B samples where to evaluate the score.
        
    data_samples : [n, C, H, W] pyTorch tensor
        Contains the n training samples of size CxHxW.
        
    t : int
        Timestep in [0, T].

    Returns
    -------
    Femp : [B, C, H, W] pyTorch tensor
        Contains the empirical score estimated at points 'samples'
        using training data 'data_samples'
    '''
    n = len(data_samples)
    d = int(np.prod(data_samples.shape[1::]))
    n_gen = len(samples)
    dim = len(data_samples.shape)
    Dt = 1 - np.exp(-2*t)
     
    Xs = data_samples*np.exp(-t)
    Xs = Xs.reshape(-1, d).to(device)
    s = samples.reshape(-1, 1, d).to(device)
    
    # Tensorized version (12/09/24)
    dist = torch.norm(s - Xs, dim=2, p=2)
    
    dst_sq = dist**2
    mindist = dst_sq.min(1).values.reshape(-1, 1)
    sumexp = torch.exp(-(dst_sq-mindist)/2/Dt).sum(1).reshape(-1, 1)
    log_Pt = -mindist/2/Dt + torch.log(sumexp) - np.log(n) - d/2*np.log(2*np.pi*Dt)
    
    # Compute derivative of the probability
    diff = s - Xs
    
    weighted_sumexp = (diff*torch.exp(-(dst_sq-mindist)/2/Dt).reshape(n_gen, n, -1)).sum(1)
    signs = (torch.sign(weighted_sumexp) + 1) / 2 # To have 0 and 1
    log_sum = torch.log(torch.abs(-weighted_sumexp)) + 1j*np.pi*signs
    log_Ptprime = - np.log( n * Dt * (2*np.pi*Dt)**(d/2) ) - mindist/2/Dt + log_sum
    
    Femp = torch.exp(log_Ptprime - log_Pt).real
    Femp = Femp.reshape(samples.shape)
    
    return Femp

def compute_logPt_emp(samples, data_samples, t, device='cpu'):
    '''
    Parameters
    ----------
    samples : [B, C, H, W] pyTorch tensor
        Contains the B samples where to evaluate the probability.
        
    data_samples : [n, C, H, W] pyTorch tensor
        Contains the n training samples of size CxHxW.
        
    t : int
        Timestep in [0, T].

    Returns
    -------
    Pemp : [B, C, H, W] pyTorch tensor
        Contains the empirical probability estimated at points 'samples'
        using training data 'data_samples'
    '''
    n = len(data_samples)
    d = int(np.prod(data_samples.shape[1::]))
    n_gen = len(samples)
    dim = len(data_samples.shape)
    Dt = 1 - np.exp(-2*t)
     
    Xs = data_samples*np.exp(-t)
    Xs = Xs.reshape(-1, d).to(device)
    s = samples.reshape(-1, 1, d).to(device)
    
    # Tensorized version (12/09/24)
    dist = torch.norm(s - Xs, dim=2, p=2)
    
    dst_sq = dist**2
    mindist = dst_sq.min(1).values.reshape(-1, 1)
    sumexp = torch.exp(-(dst_sq-mindist)/2/Dt).sum(1).reshape(-1, 1)
    log_Pt = -mindist/2/Dt + torch.log(sumexp) - np.log(n) - d/2*np.log(2*np.pi*Dt)
    
    return log_Pt

def compute_logPt_gen(x, mu, sig, t, device='cpu'):
    dt = 1 - np.exp(-2*t)
    Gamma_t = sig**2*np.exp(-2*t) + dt
    d = x.shape[1]
    mean = torch.ones_like(x, device=device)*mu*np.exp(-t)
    norm = (2*np.pi*Gamma_t)**(-d/2)*((sig**2)**d)**(-1/2) * 1/2
    maha1 = -torch.norm(x-mean, dim=1)**2/2/Gamma_t
    maha2 = -torch.norm(x+mean, dim=1)**2/2/Gamma_t
    prob = norm * (torch.exp(maha1) + torch.exp(maha2))
    return torch.log(prob.unsqueeze(1))

# Score model
def compute_score_model(model, x, t):
    sqrt_dt = np.sqrt(1 - np.exp(-2*t))
    return -model(x) / sqrt_dt

def compute_score_model_alpha(model, x, t, df):
    ts = torch.ones(size=(x.shape[0],), device=x.device, dtype=torch.long) * t
    sqrt_dt = df.sqrt_one_minus_alpha_cumulative[t].item() # Error TB corrected 07/03/25
    return -model(x, ts) / sqrt_dt
 
# Ratio
def compute_ratio_gen_mem(gen_score, mem_score, model_score):
    e_gen = torch.linalg.norm(model_score - gen_score, 2, axis=1).mean().item()
    e_mem = torch.linalg.norm(gen_score - mem_score, 2, axis=1).mean().item()
    R = e_gen / e_mem
    return R

def compute_ratio_gen_mem2(gen_score, mem_score, model_score):
    e_gen = torch.linalg.norm(model_score - gen_score, 2, axis=1).mean().item()
    e_mem = torch.linalg.norm(model_score - mem_score, 2, axis=1).mean().item()
    e_gen_mem = torch.linalg.norm(gen_score - mem_score, 2, axis=1).mean().item()
    R = (e_mem - e_gen) / e_gen_mem
    return R



# TB 250317: OK (checked vs numpy probabilities)
def compute_logPt_gen_alpha(x, mu, sig, t, df, device='cpu'):
    dt = (1 - df.alpha_cumulative[t])
    Gamma_t = sig**2*df.alpha_cumulative[t] + dt
    d = x.shape[1]
    mean = torch.ones_like(x, device=device)*mu*df.sqrt_alpha_cumulative[t]
    norm = (2*np.pi*Gamma_t)**(-d/2)*((sig**2)**d)**(-1/2) * 1/2
    maha1 = -torch.norm(x-mean, dim=1)**2/2/Gamma_t
    maha2 = -torch.norm(x+mean, dim=1)**2/2/Gamma_t
    prob = norm * (torch.exp(maha1) + torch.exp(maha2))
    return torch.log(prob.unsqueeze(1))


def compute_generalization_score_GMM_alpha(x, mu, t, df, device='cpu'):
    '''
    Compute the population score of a 2-Gaussian Mixture Model
    Assuming P0 centred on +mu and -mu, and unit variance.
    '''
    d = x.shape[1]
    mean = (torch.ones(d, device=device)*mu)
    m = x @ (mean * torch.sqrt(df.alpha_cumulative[t]))
    mean = mean.expand(x.shape)
    return (mean*torch.sqrt(df.alpha_cumulative[t]) * torch.tanh(m).reshape(-1, 1) - x)


def compute_empirical_score_alpha(samples, data_samples, t, df, device='cpu'):
    '''
    Parameters
    ----------
    samples : [B, C, H, W] pyTorch tensor
        Contains the B samples where to evaluate the score.
        
    data_samples : [n, C, H, W] pyTorch tensor
        Contains the n training samples of size CxHxW.
        
    t : int
        Timestep in [0, T].

    Returns
    -------
    Femp : [B, C, H, W] pyTorch tensor
        Contains the empirical score estimated at points 'samples'
        using training data 'data_samples'
    '''
    n = len(data_samples)
    d = int(np.prod(data_samples.shape[1::]))
    n_gen = len(samples)
    dim = len(data_samples.shape)
    Dt = 1 - df.alpha_cumulative[t].cpu().item()
     
    ts = torch.ones(size=(n,), device=device,
                    dtype=torch.long) * t
    Xs = Diffusion.get(df.sqrt_alpha_cumulative, t=ts, dim=2) * data_samples
    Xs = Xs.reshape(-1, d).to(device)
    s = samples.reshape(-1, 1, d).to(device)
    
    # Tensorized version (12/09/24)
    dist = torch.norm(s - Xs, dim=2, p=2)
    
    dst_sq = dist**2
    mindist = dst_sq.min(1).values.reshape(-1, 1)
    sumexp = torch.exp(-(dst_sq-mindist)/2/Dt).sum(1).reshape(-1, 1)
    log_Pt = -mindist/2/Dt + torch.log(sumexp) - np.log(n) - d/2*np.log(2*np.pi*Dt)
    
    # Compute derivative of the probability
    diff = s - Xs
    
    weighted_sumexp = (diff*torch.exp(-(dst_sq-mindist)/2/Dt).reshape(n_gen, n, -1)).sum(1)
    signs = (torch.sign(weighted_sumexp) + 1) / 2 # To have 0 and 1
    log_sum = torch.log(torch.abs(-weighted_sumexp)) + 1j*np.pi*signs
    log_Ptprime = - np.log(n * Dt) - (d/2)*np.log(2*np.pi*Dt) - mindist/2/Dt + log_sum
    
    Femp = torch.exp(log_Ptprime - log_Pt).real
    Femp = Femp.reshape(samples.shape)
    
    return Femp
