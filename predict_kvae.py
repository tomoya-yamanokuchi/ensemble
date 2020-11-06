import json
import os
import pickle
import time
import copy
import subprocess
import numpy as np
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
        print()

    
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
        dim_y = config.dim_outputs

        with open(path_conf + "/norm_info.pickle", "rb") as f: 
            norm_info = pickle.load(f)
        self.norm_info = norm_info

        x_train = norm_dnn_data.normalize_x(x_train, norm_info)
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


        self.predict_test1(mean_var=True)
        self.predict_test1(mean_var=False)
        self.predict_test_train(mean_var=True)
        self.predict_test_train(mean_var=False)
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
            fig, ax = plt.subplots()

            if mean_var is False: 
                for m in range(N_ensemble):
                    ax.plot(y_predict[n, :, 0, m], color="r")
            else: 
                mean  = np.mean(y_predict[n, :, 0, :], axis=-1)
                std   = np.std( y_predict[n, :, 0, :], axis=-1)
                lower = mean - 2.0*std
                upper = mean + 2.0*std
                x = range(self.step)
                color_fill = "skyblue"
                ax.fill_between(x, lower, upper, alpha=0.6, color=color_fill)
                ax.plot(x, mean,  "-",  color="b")

            ax.plot(y_test[n, :, 0], color="k")
            plt.show()



    def predict_test2(self): 
        # self.x_train.reshape(-1, self.dim_x)[:, 0]
        N_mesh = 100
        x_test = np.tile(np.linspace(-0.5, 1.5, N_mesh).reshape(-1, 1), (1, self.dim_x))

        X1, X2 = np.meshgrid(x_test[:, 0], x_test[:, 1])
        X_TEST = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=-1)

        ind_alpha  = -100
        N_zu = 100
        zu_add     = np.tile(self.x_train[ind_alpha, 4, :2].reshape(1, 2), (N_zu, 1))
        X_TEST_add = np.tile(self.x_train[ind_alpha, 4, -3:].reshape(1, -1), (N_mesh**2, 1))
        X_TEST     = np.concatenate([X_TEST, X_TEST_add], axis=-1)

        N_inputs, _ = X_TEST.shape
        N_ensemble = self.N_ensemble

        y_predict = self.model.predict(X_TEST)
        y_predict = np.stack(y_predict, axis=-1)

        fig = plt.figure(figsize=(8,6))
        ax3d = plt.axes(projection="3d")
        ax3d = plt.axes(projection='3d')

        yy = np.minimum(1,  y_predict)
        yy = np.maximum(-1, y_predict)
        # for m in range(N_ensemble): 
        for m in range(1): 
            ax3d.plot_surface(X1, X2, yy[:, 0, m].reshape(N_mesh, N_mesh),cmap='plasma')
        ax3d.plot(zu_add[:, 0], zu_add[:, 1], np.linspace(-1.2, 1.2, N_zu),  markersize=30, color="r")
        ax3d.set_title('Surface Plot in Matplotlib')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_zlim([-1.2, 1.2])

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