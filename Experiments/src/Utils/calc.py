import numpy as np

def unique_modulus(a, mod):
    return np.unique(a - a%mod)