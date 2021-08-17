import os
import time
import numpy as np
from omegaconf import OmegaConf, DictConfig

class Repository: 
    def save_config(self, config: DictConfig):
        assert type(config) == DictConfig, "expected: DictConfig, input: {}".format(type(config))
        config.log_dir = "/hdd_mount/ensemble/logs"
        run_name       = "M{}_".format(config.N_ensemble) + config.kvae_model + "_" + time.strftime('%Y%m%d%H%M%S', time.localtime())
        config.log_dir = os.path.join(config.log_dir, run_name)
        os.makedirs(config.log_dir, exist_ok=True)
        OmegaConf.save(config, config.log_dir + "/config.yaml")
        

    def load_dataset(self, path): 
        npzfile  = np.load(path)
        y_train  = npzfile['z_true'].astype(np.float32)
        x_train1 = npzfile['z'].astype(np.float32)
        x_train2 = npzfile['u'].astype(np.float32)
        # x_train3 = npzfile['alpha'].astype(np.float32)
        # x_train  = np.concatenate([x_train1, x_train2, x_train3], axis=-1)
        x_train  = np.concatenate([x_train1, x_train2], axis=-1)
        # return x_train[:1], y_train[:1]
        return x_train, y_train
    
    
    def save_norm_data(self, path, x_min, x_max, y_min, y_max):
        np.savez(path, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        

    def save_norm_data_z_score(self, path, y_log_mean, y_log_std):
        np.savez(path, y_log_mean=y_log_mean, y_log_std=y_log_std)
    
        
    def load_norm_data(self, path): 
        npzfile = np.load(path)
        y_min = npzfile["y_min"]
        y_max = npzfile["y_max"]
        return y_min, y_max
    
    def load_norm_data_z_score(self, path): 
        npzfile    = np.load(path)
        y_log_mean = npzfile["y_log_mean"]
        y_log_std  = npzfile["y_log_std"]
        return y_log_mean, y_log_std