import numpy as np

def normalize_y(y, norm_info): 
    y = np.log(y + norm_info["bias"])
    y = (y - norm_info["y_log_mean"]) / norm_info["y_log_std"]
    return y

def denormalize_y(y, norm_info):  
    y = y * norm_info["y_log_std"] + norm_info["y_log_mean"]
    y = np.exp(y) - norm_info["bias"]
    return y

def normalize_x(x, norm_info, specify_dim=None): 
    N, dim_x = x.shape

    if specify_dim is None: 
        x_min = norm_info["x_min"].reshape(1, dim_x)
        x_max = norm_info["x_max"].reshape(1, dim_x)
    else: 
        x_min = norm_info["x_min"][specify_dim]
        x_max = norm_info["x_max"][specify_dim]

    x = (x - x_min) / (x_max - x_min)
    return x


def denormalize_x(x, norm_info, specify_dim=None): 
    N, dim_x = x.shape
    if specify_dim is None: 
        x_min = norm_info["x_min"].reshape(1, dim_x)
        x_max = norm_info["x_max"].reshape(1, dim_x)
    else: 
        x_min = norm_info["x_min"][specify_dim]
        x_max = norm_info["x_max"][specify_dim]

    x = x * (x_max - x_min) + x_min
    return x