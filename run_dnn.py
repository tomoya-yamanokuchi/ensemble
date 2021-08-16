import json
import os
import pickle
import time
import copy
from scipy import stats
import subprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig
from mpl_toolkits.mplot3d import Axes3D
import myloss
import PlotHandler as plothandler

from dnn_kvae import DNNModel
# from config_kvae import reload_config, get_image_config
from config_seesaw import reload_config, get_image_config
from myCallBack import MYCallBack
from Repository import Repository
import ConsoleOutput  as cout
import Normalization as norm



class RUN_DNN:
    def run(self, config):     
        self.config = config   
        repository  = Repository()
        repository.save_config(config)
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
        
        x_train, y_train     = repository.load_dataset(config.dataset)
        N_train, step, dim_x = x_train.shape
        dim_y                = y_train.shape[-1]
        y_train_origin       = copy.deepcopy(y_train)
        
        # self.plot_all_sequence(x_train[:, :, -8:])
        # self.plot_all_sequence(y_train[:, :, :])

        cout.console_output(N_train, step, dim_x, dim_y)

        # -------------- normalize and convert -------------------
        # x_train, x_min , x_max = norm.normalize(x_train)
        # y_train                = np.log(y_train)
        # y_train, y_min , y_max = norm.normalize(y_train)
        y_train, y_log_mean, y_log_std = norm.normalize_z_score(y_train)
        # repository.save_norm_data(config.log_dir + "/norm_data.npz",  x_min, x_max, y_min, y_max)
        repository.save_norm_data_z_score(config.log_dir + "/norm_data.npz",  y_log_mean, y_log_std)
        
        # plothandler.plot_all_sequence(x_train[:, :, :])
        plothandler.plot_all_sequence(y_train[:, :, :])

        # y_restore = np.exp(norm.denormalize(y_train, y_min, y_max))
        # # self.plot_all_sequence(y_restore)
        # error = np.linalg.norm(y_restore - y_train_origin)

        
        # -------------- split data -------------------
        # x_train, x_valid = train_test_split(x_train, test_size=0.1)
        # y_train, y_valid = train_test_split(y_train, test_size=0.1)


        # -------------- split data -------------------
        checkpoint_path = config.log_dir + "/cp-{epoch:04d}.ckpt"
        checkpoint_dir  = os.path.dirname(checkpoint_path)
        cp_callback     = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                        save_weights_only=True,
                                                        verbose=0,
                                                        period=config.epoch)

        mycb = MYCallBack(config.log_dir)


        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            with tf.variable_scope("ensemble"):

                dnn       = DNNModel(config)
                model     = dnn.nn_ensemble(N_ensemble=config.N_ensemble)
                optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
                model.compile(loss=myloss.smooth_L1, optimizer=optimizer, metrics=['mse'])
                # model.compile(loss='msle', optimizer=optimizer, metrics=['msle'])
                model.summary()
                

                saver = tf.train.Saver()
                model.fit(
                    x                   = x_train.reshape(-1, dim_x), 
                    # y                   = [y_train[:,:,0].reshape(-1)]*config.N_ensemble, 
                    y                   = [y_train.reshape(-1, dim_y)]*config.N_ensemble, 
                    epochs              = config.epoch,
                    batch_size          = config.batch_size,  
                    # validation_data     = (x_train.reshape(-1, dim_x), [y_train[:,:,0].reshape(-1)]*config.N_ensemble), 
                    validation_data     = (x_train.reshape(-1, dim_x), [y_train.reshape(-1, dim_y)]*config.N_ensemble), 
                    callbacks           = [mycb], 
                    use_multiprocessing = True
                )

                # model.save(checkpoint_dir + "/model.h5")
                saver.save(sess, checkpoint_dir + '/model.ckpt')

        print("end")
        
        

if __name__ == "__main__":
    import hydra
    from attrdict import AttrDict
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.config_store import ConfigStore
    from hydra.experimental import (
        initialize,
        initialize_config_module,
        initialize_config_dir,
        compose,
    )
    
    @hydra.main(config_path="conf/config_ICRA2022.yaml")
    def get_config(cfg: DictConfig) -> None:
        
        run = RUN_DNN()
        run.run(cfg)
    
    get_config()
    
    
