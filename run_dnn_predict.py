from numpy import random
from Repository import Repository
import json
import os
import pickle
import time
import copy
from omegaconf.omegaconf import OmegaConf
import seaborn as sns
import subprocess
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
from scipy import stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
# from config_kvae import reload_config, get_image_config
from config_seesaw import reload_config, get_image_config
from dnn_kvae import DNNModel
import normalize_dnn_data as norm_dnn_data
import Normalization as norm
import ConsoleOutput as cout
import PlotHandler as plothandler

class RUN_PREDICT:
    def plot_hist(self, x): 
        fig, ax = plt.subplots()
        plt.hist(x)
        print("======================")
        print('       var: {0:.2f}'.format(np.var(x)))
        print('       std:{0:.2f}'.format(np.std(x)))
        print('      skew: {0:.2f}'.format(stats.skew(x)))
        print('  kurtosis: {0:.2f}'.format(stats.kurtosis(x)))
        print("======================")
        plt.show()


    def plot_line(self, x): 
        seq, step = x.shape
        fig, ax = plt.subplots()
        for s in range(seq): 
            ax.plot(x[s, :])
        plt.show()
        print()

    def plot_DataFrame(self, x, ylabel="ylabel", figsize=(10, 5)):
        # x = np.random.randn(10, 30, 4)
        x = np.transpose(x, (1, 0, 2))
        step, N, dim_x = x.shape
        sns.set()

        fig, ax = plt.subplots(nrows=1, ncols=dim_x, figsize=figsize)
        for d in range(dim_x): 
            df = pd.DataFrame(x[:,:,d], columns=range(N))
            df.cumsum()
            if dim_x == 1: 
                df.plot(ax=ax, legend=False)
            else: 
                df.plot(ax=ax[d], legend=False)

        ax.set_xlim([0, step-1])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("step")
        plt.show()

        # for d in range(dim_x): 
        #     df = pd.DataFrame(x[:,:,d], columns=range(N))
        #     df.cumsum().plot()
        #     plt.show()

    
    def plot_3d(self, x, y, z):
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, projection='3d')
        self.axis.set_xlabel('xlabel')
        self.axis.set_ylabel('ylabel')
        self.axis.set_zlabel('zlabel')

        self.axis.plot_surface(x, y, z)

        plt.show()


    def add_noise(self, x, dim_list, std, num): 
        assert len(x.shape) == 3
        sequence, step, dim = x.shape
        xx = np.zeros([num, step, dim])
        for ind, d in enumerate(range(dim)): 
            if d in dim_list: 
                xx[:, :, d] = x[:, :, d] + np.random.randn(*xx[:, :, d].shape)*std
            else: 
                xx[:, :, d] = x[:, :, d]
        return xx



    def run(self, config): 
        self.config = config
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

        repository = Repository()
        x_train, y_train     = repository.load_dataset(config.dataset)
        
        # --------------------
        # x_aux = self.add_noise(x_train, dim_list=[6, 7, 8, 9, 10,11,12,13], std=1, num=20)
        # x_train = np.concatenate((x_train, x_aux), axis=0)
        # y_train = np.tile(y_train, (x_train.shape[0], 1, 1))
        # --------------------
        
        
        N_train, step, dim_x = x_train.shape
        dim_y                = y_train.shape[-1]
        y_train_origin       = copy.deepcopy(y_train)
        

        plothandler.plot_all_sequence(x_train[:, :, :])
        # self.plot_all_sequence(y_train[:, :, :])

        cout.console_output(N_train, step, dim_x, dim_y)
        
        # y_min, y_max  = repository.load_norm_data(self.config.load_dir + "/norm_data.npz")
        # y_train       = np.log(y_train)
        
        # y_log_mean, y_log_std = repository.load_norm_data_z_score(self.config.load_dir + "/norm_data.npz")
        # y_train, _, _ = norm.normalize_z_score(y_train, y_log_mean, y_log_std)
        
        # plothandler.plot_all_sequence(y_train[:, :, :])

        dnn = DNNModel(config)

        # session_config = tf.compat.v1.ConfigProto()
        # session_config.gpu_options.allow_growth = True
        # self.dnn_model = tf.keras.models.load_model( path_conf + "/model.h5", custom_objects={'swish': dnn.swish })
        # saver = tf.train.Saver()
        sess = tf.keras.backend.get_session()
        # build model
        with tf.variable_scope("ensemble"):
            dnn = DNNModel(config)
            # self.dnn_model_instance = tf.keras.models.load_model( path_conf + "/model.h5", custom_objects={'swish': dnn.swish })
            self.dnn_model = dnn.nn_ensemble(N_ensemble=self.config.N_ensemble)

        saver_dnn = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ensemble'))
        saver_dnn.restore(sess, config.reload_model)


        self.N_train = N_train
        self.step    = step
        self.dim_x   = dim_x
        self.dim_y   = dim_y
        self.x_train = x_train
        self.y_train = y_train
        

        y_predict = self.dnn_model.predict(x_train.reshape(-1, self.dim_x))
        y_predict = np.stack(y_predict, axis=-1).reshape(N_train, self.step, self.dim_y, config.N_ensemble)

        plothandler.predict_both(x_train, y_train, x_train, y_train, y_predict, N_test=20)
        
        
        # self.predict_test1(mean_var=True)
        # self.predict_test1(mean_var=False)
        self.predict_test_train(mean_var=True)
        # self.predict_test_train(mean_var=False)
        # self.predict_test2()

        # for n in range(N_test):
        #     fig, ax = plt.subplots()
        #     yp = np.zeros([N_ensemble, step])
        #     for m in range(N_ensemble):
        #         y = y_predict[m].reshape(N_test, -1, config.dim_outputs)
        #         y = y[n, :, 0]
        #         yp[m] = copy.deepcopy(y)

        #     for m in range(N_ensemble):
        #         ax.plot(yp[m, :], color="r")
        #     ax.plot(y_test[n, :, 0], color="k")
        #     plt.show()


        # self.plot_3d(X1, X2, yy)



    def predict_test1(self, mean_var=False): 
        N_ensemble = self.config.N_ensemble
        N_test = 15

        ind_test = np.linspace(0, self.N_train-1, N_test, dtype=int)
        x_test = self.x_train[ind_test]
        y_test = self.y_train[ind_test]

        y_predict = self.dnn_model.predict(x_test.reshape(-1, self.dim_x))
        y_predict = np.stack(y_predict, axis=-1).reshape(N_test, self.step, self.dim_y, N_ensemble)

        for n in range(N_test):
            fig, ax = plt.subplots(figsize=(3, 3))

            if mean_var is False: 
                for m in range(N_ensemble):
                    ax.plot(y_predict[n, :, 0, m], color="mediumvioletred")
            else: 
                mean  = np.mean(y_predict[n, :, 0, :], axis=-1)
                std   = np.std( y_predict[n, :, 0, :], axis=-1)
                lower = mean - 2.0*std
                upper = mean + 2.0*std
                x = range(self.step)
                color_fill = "thistle"
                ax.fill_between(x, lower, upper, alpha=0.6, color=color_fill)
                ax.plot(x, mean,  "-",  color="mediumvioletred")

            ax.plot(y_test[n, :, 0], color="k")
            plt.show()





    def predict_test2(self): 
        # self.x_train.reshape(-1, self.dim_x)[:, 0]

        ind_alpha  = -100
        N_zu = 100
        zu_add = np.tile(self.x_train[ind_alpha, 4, :2].reshape(1, 2), (N_zu, 1))
        zu = zu_add[0]
        print("zu: ", zu)

        N_mesh = 100
        x1 = np.linspace(zu[0] - zu[0]*0.01, zu[0] + zu[0]*0.01, N_mesh)
        x2 = np.linspace(zu[1] - zu[1]*0.01, zu[1] + zu[1]*0.01, N_mesh)
        X1, X2 = np.meshgrid(x1, x2)
        X_TEST = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=-1)

        X_TEST_add = np.tile(self.x_train[ind_alpha, 4, -3:].reshape(1, -1), (N_mesh**2, 1))
        X_TEST     = np.concatenate([X_TEST, X_TEST_add], axis=-1)

        N_inputs, _ = X_TEST.shape
        N_ensemble = self.config.N_ensemble

        y_predict = self.dnn_model.predict(X_TEST)
        y_predict = np.stack(y_predict, axis=-1)

        # clip
        clip_max_y = 1.1
        clip_min_y = 0.7
        y_predict = np.minimum(clip_max_y, y_predict)
        y_predict = np.maximum(clip_min_y, y_predict)

        fig = plt.figure(figsize=(8,6))
        ax3d = plt.axes(projection="3d")
        ax3d = plt.axes(projection='3d')

        # yy = np.minimum(1.2,  y_predict)
        # yy = np.maximum(0.5, y_predict)
        # for m in range(N_ensemble): 
        for m in range(1): 
            ax3d.plot_surface(X1, X2, y_predict[:, 0, m].reshape(N_mesh, N_mesh),cmap='plasma')
        # ax3d.plot(zu_add[:, 0], zu_add[:, 1], np.linspace(-1.2, 1.2, N_zu),  markersize=30, color="r")
        ax3d.set_title('Surface Plot in Matplotlib')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        # ax3d.set_zlim([-1.2, 1.2])


        x = zu[0]
        y = x2
        z = np.linspace(clip_min_y, clip_max_y, N_mesh)
        Y,Z = np.meshgrid(y,z)
        X = np.array([x]*Y.shape[0])
        ax3d.plot_surface(X,Y,Z,alpha=0.3)

        plt.show()



    def predict_test_train(self, mean_var=True):
        N_ensemble = self.config.N_ensemble
        x_test = self.x_train
        y_test = self.y_train

        y_predict = self.dnn_model.predict(x_test.reshape(-1, self.dim_x))
        y_predict = np.stack(y_predict, axis=-1)

        y_mean = np.mean(y_predict, axis=-1)



        y_true = y_test.reshape(-1, self.dim_y) 
        
        y_mean    = norm_dnn_data.denormalize_y(y_mean,    self.norm_info)
        y_true    = norm_dnn_data.denormalize_y(y_true,    self.norm_info)
        y_predict = norm_dnn_data.denormalize_y(y_predict, self.norm_info)
        
        y_min = np.concatenate([y_mean, y_true], axis=-1).reshape(-1).min()
        y_max = np.concatenate([y_mean, y_true], axis=-1).reshape(-1).max()

        error = (y_true - y_mean)**2

        if mean_var is True: 
            fig, ax = plt.subplots()
            ax.plot(y_true, y_mean,'b.', markersize=3)
            ax.plot([y_min, y_max],[y_min, y_max], color="k")
            ax.set_xlabel('y_true')
            ax.set_ylabel('y_predict')
            plt.show()
        else: 
            fig, ax = plt.subplots()
            for i in range(N_ensemble): 
                ax.plot(y_true, y_predict[:, 0, i], linestyle="None", marker=".", color=cm.hsv(i/N_ensemble), markersize=3)
            ax.plot([y_min, y_max],[y_min, y_max], color="k")
            ax.set_xlabel('y_true')
            ax.set_ylabel('y_predict')
            plt.show()



if __name__ == "__main__":
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816164519"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816171605"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816172023"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816172701"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816174824"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816175115"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816202456"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816203225"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816203544"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816203825"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816204000"
    
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210816221551"
    path = "M5_dclaw_64x64_N151_seq20_dim_a8_Epoch10000_seed1_SCREW_NO_RANDOMIZE_20210813185936_canonical_to_canonical_kvae_20210817002742"
    
    path = "/hdd_mount/ensemble/logs/" + path
    config              = OmegaConf.load(path + "/config.yaml")
    config.load_dir     = path
    config.reload_model = path + "/model.ckpt"

    run = RUN_PREDICT()
    run.run(config)