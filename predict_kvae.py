import json
import os
import pickle
import time
import copy
import seaborn as sns
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from config_kvae import reload_config, get_image_config
from dnn_kvae import DNNModel
import normalize_dnn_data as norm_dnn_data


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

    def plot_DataFrame(self, x):
        # x = np.random.randn(10, 30, 4)
        x = np.transpose(x, (1, 0, 2))
        step, N, dim_x = x.shape
        sns.set()

        fig, ax = plt.subplots(nrows=1, ncols=dim_x, figsize=(9, 6))
        for d in range(dim_x): 
            df = pd.DataFrame(x[:,:,d], columns=range(N))
            df.cumsum()
            df.plot(ax=ax[d], legend=False)
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


    def run(self): 
        path_conf = "./logs/N_ensemble5_20201105035501"
        # path_conf = "./logs/N_ensemble5_20201105035349"
        path_conf = "./logs/N_ensemble1_20201106171316"
        path_conf = "./logs/N_ensemble1_20201109212153"
        path_conf = "./logs/N_ensemble5_20201110062703"
        path_conf = "./logs/N_ensemble5_20201110062835"

        config = get_image_config()
        config.FLAGS.reload_model = path_conf + "/"
        config = reload_config(config.FLAGS)
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

        dnn = DNNModel(config)
        self.model = tf.keras.models.load_model( path_conf + "/model.h5", custom_objects={'swish': dnn.swish })

        N_ensemble = config.N_ensemble
        self.N_ensemble = N_ensemble

        # =================
        #   training
        # =================
        npzfile  = np.load(config.dataset)
        y_train  = npzfile['pred_error'].astype(np.float32)
        x_train1 = npzfile['z'].astype(np.float32)
        x_train2 = npzfile['u'].astype(np.float32)
        x_train3 = npzfile['alpha'].astype(np.float32)
        x_train  = np.concatenate([x_train1, x_train2, x_train3], axis=-1)

        N_train, step, dim_x = x_train.shape

        self.plot_DataFrame(x_train1)

        # fig, ax = plt.subplots(figsize=(6, 5))
        # for d in range(dim_x): 
        #     plt.cla()

        # for i in range(N_train):
        #     ax.plot(x_train[i, :, 5])
        # plt.show()

        # for i in range(dim_x): 
        #     self.plot_hist(x_train.reshape(-1, dim_x)[:, i])

        # for i in range(y_train.shape[-1]): 
        #     self.plot_line(y_train[:, :, i])


        y_train = np.expand_dims(np.sum(y_train, axis=-1), axis=-1)
        N_train, step, dim_y = y_train.shape

        print("=====================")
        print("   N_train : ", N_train) 
        print("      step : ",   step)
        print("     dim_x : ",  dim_x)
        print("     dim_y : ",  dim_y)
        print("=====================")

        with open(path_conf + "/norm_info.pickle", "rb") as f: 
            norm_info = pickle.load(f)
        self.norm_info = norm_info

        x_train = norm_dnn_data.normalize_x(x_train.reshape(-1, dim_x), norm_info)
        x_train = x_train.reshape(N_train, step, dim_x)

        y_train = norm_dnn_data.normalize_y(y_train, norm_info)


        # fig, ax = plt.subplots()
        # for i in range(N_train):
        #     ax.plot(y_train[i, :, 0])
        # plt.show()

        self.N_train = N_train
        self.step    = step
        self.dim_x   = dim_x
        self.dim_y   = dim_y
        self.x_train = x_train
        self.y_train = y_train

        self.predict_both(N_test=6)
        # self.predict_test1(mean_var=True)
        # self.predict_test1(mean_var=False)
        self.predict_test_train(mean_var=True)
        self.predict_test_train(mean_var=False)
        self.predict_test2()

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


        self.plot_3d(X1, X2, yy)



    def predict_test1(self, mean_var=False): 
        N_ensemble = self.N_ensemble
        N_test = 15

        ind_test = np.linspace(0, self.N_train-1, N_test, dtype=int)
        x_test = self.x_train[ind_test]
        y_test = self.y_train[ind_test]

        y_predict = self.model.predict(x_test.reshape(-1, self.dim_x))
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


    def predict_both(self, N_test): 
        N_ensemble = self.N_ensemble
        # N_test = 3

        ind_test = np.linspace(0, self.N_train-1, N_test, dtype=int)
        x_test = self.x_train[ind_test]
        y_test = self.y_train[ind_test]

        y_predict = self.model.predict(x_test.reshape(-1, self.dim_x))
        y_predict = np.stack(y_predict, axis=-1).reshape(N_test, self.step, self.dim_y, N_ensemble)

        fig, ax = plt.subplots(2, N_test, figsize=(9, 6))
        for n in range(N_test):
            for m in range(N_ensemble):
                ax[0, n].plot(y_predict[n, :, 0, m], color="mediumvioletred", label="DNN predict")
            ax[0, n].plot(y_test[n, :, 0], color="k", label="Ground Truth")

            mean  = np.mean(y_predict[n, :, 0, :], axis=-1)
            std   = np.std( y_predict[n, :, 0, :], axis=-1)
            lower = mean - 2.0*std
            upper = mean + 2.0*std
            x = range(self.step)
            color_fill = "thistle"
            ax[1, n].fill_between(x, lower, upper, alpha=0.6, color=color_fill)
            ax[1, n].plot(x, mean,  "-",  color="mediumvioletred", label="DNN predict")
            ax[1, n].plot(y_test[n, :, 0], color="k", label="Ground Truth")

            ax[1, n].set_xlabel("Step", fontsize=18)
            if n == 0: 
                ax[0, n].set_ylabel(r"$ e_{t_+1} $", fontsize=18)
                ax[1, n].set_ylabel(r"$ e_{t_+1} $", fontsize=18)

        lines = []
        labels = []
        for _ax in fig.axes:
            axLine, axLabel = _ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)
        # fig.legend(lines, labels[:5], bbox_to_anchor=(0.75, 0.95,), ncol=5, fontsize=16)
        fig.legend(lines[5:7], labels[5:7], loc="upper center", ncol=2, fontsize=14)

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
        N_ensemble = self.N_ensemble

        y_predict = self.model.predict(X_TEST)
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
        N_ensemble = self.N_ensemble
        x_test = self.x_train
        y_test = self.y_train

        y_predict = self.model.predict(x_test.reshape(-1, self.dim_x))
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
    
    run = RUN_PREDICT()
    run.run()