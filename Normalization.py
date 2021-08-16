import numpy as np


def normalize(x, x_min=None, x_max=None):
    assert len(x.shape) == 3 
    dim          = x.shape[-1]
    
    if (x_min is None) and (x_max is None): 
        x_max = np.max(np.max(x, axis=0), axis=0).reshape(1, 1, dim)
        x_min = np.min(np.min(x, axis=0), axis=0).reshape(1, 1, dim)
    
    x_normalized = (x - x_min) / (x_max - x_min)
    return x_normalized, x_min , x_max


def normalize_z_score(x, x_log_mean=None, x_log_std=None):
    assert len(x.shape) == 3 
    dim = x.shape[-1]
    
    if (x_log_mean is None) and (x_log_std is None): 
        x_log_mean   = np.mean(x.reshape(-1, dim), axis=0).reshape(1, 1, dim)
        x_log_std    = np.std(x.reshape(-1, dim), axis=0).reshape(1, 1, dim)
        
    x_normalized = (x - x_log_mean) / x_log_std
    return x_normalized, x_log_mean, x_log_std


def denormalize(x_normalized, x_min, x_max): 
    assert len(x_normalized.shape) == 3 
    assert len(x_min.shape) == 3 
    assert len(x_max.shape) == 3 
    x = x_normalized * (x_max - x_min) + x_min
    return x
    