import os
import time
import pickle
import numpy as np
from omegaconf import OmegaConf, DictConfig

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

import sys
import pathlib
p = pathlib.Path(__file__).resolve()
sys.path.append(str(p.parent))



class Repository:
    def save_config(self, config: DictConfig):
        assert type(config) == DictConfig, "expected: DictConfig, input: {}".format(type(config))

        if config.dataset == "kvae":    self.log_dir = config.log_dir
        else:                           self.log_dir = str(p.parent) + "/logs/M{}_{}".format(config.N_ensemble, time.strftime('%Y%m%d%H%M%S', time.localtime()))
        os.makedirs(self.log_dir, exist_ok=True)
        OmegaConf.save(config, self.log_dir + "/config.yaml")


    def load_dataset(self, path):
        N_valid = 500
        x_valid1 = np.linspace(-np.pi*2.0, -np.pi, int(N_valid*0.5))
        x_valid2 = np.linspace( np.pi,  np.pi*2.0, int(N_valid*0.5))
        x_valid  = np.hstack([x_valid1, x_valid2])

        y_mean  = np.sin(x_valid)
        y_valid = y_mean + np.random.randn(N_valid) * np.std(4*0.225*np.abs(np.sin(1.5 * x_valid + np.pi/8.0)))

        plt.plot(x_valid, y_valid, "x", color="m")
        plt.show()


        return x_valid, y_valid
        # return x_train[:1], y_train[:1]




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


    def reload_config(self, config):
        if (config.reload_model != '') or (config.reload_model != "") or (config.reload_model is not None):
            pwd           = os.getcwd().split("outputs")[0] # This depends on hydra
            config_reload = OmegaConf.load(pwd + "logs/" + config.reload_model + "/config.yaml")
            '''
                reloadしたときに変更してもいい変数
            '''
            updatable_keys = [
                'gpu',
                'reload_model',
            ]
            for key, value in list(config.items()):
                if key in updatable_keys:
                    config_reload.__setattr__(key, value)

        return config_reload


    def save_variable_name_list(self, save_variable_name_list: list, save_path: str):
        with open(save_path + "/variable_name_list.pkl", "wb") as f:
            pickle.dump(save_variable_name_list, f)


    def load_variable_name_list(self, save_path: str):
        with open(save_path + "/variable_name_list.pkl", "rb") as f:
            variable_name_list = pickle.load(f)
        return variable_name_list
