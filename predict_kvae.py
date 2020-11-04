import json
import os
import time
import copy
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from config_kvae import reload_config, get_image_config
from dnn_kvae import DNNModel



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
        path_conf = "./logs/N_ensemble5_20201105015321"
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

        x_max = np.max(np.max(x_train, axis=0), axis=0).reshape(1, 1, dim_x)
        x_min = np.min(np.min(x_train, axis=0), axis=0).reshape(1, 1, dim_x)
        x_train = (x_train - x_min) / (x_max - x_min)

        # y_max = np.max(np.max(y_train, axis=0), axis=0).reshape(1, 1, dim_y)
        # y_min = np.min(np.min(y_train, axis=0), axis=0).reshape(1, 1, dim_y)
        # y_train = (y_train - y_min) / (y_max - y_min)
        # y_train = y_train * config.scale_inputs
        y_train = np.log(y_train)
        y_train = (y_train - np.mean(y_train.reshape(-1))) / np.std(y_train.reshape(-1))

        self.N_train = N_train
        self.step    = step
        self.dim_x   = dim_x
        self.dim_y   = dim_y
        self.x_train = x_train
        self.y_train = y_train


        self.predict_test0()
        self.predict_test1()


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



    def predict_test0(self): 
        N_ensemble = self.N_ensemble
        N_test = 15

        ind_test = np.linspace(0, self.N_train-1, N_test, dtype=int)
        x_test = self.x_train[ind_test]
        y_test = self.y_train[ind_test]

        y_predict = self.model.predict(x_test.reshape(-1, self.dim_x))

        yy = np.zeros([N_test, N_ensemble, self.step, self.dim_y])

        for n in range(N_test):
            fig, ax = plt.subplots()
            for m in range(N_ensemble):
                yy[n, m, :, :] = y_predict[m].reshape(N_test, self.step, self.dim_y)[n]

            for m in range(N_ensemble):
                ax.plot(yy[n, m, :, 0], color="r")
            ax.plot(y_test[n, :, 0], color="k")
            plt.show()



    def predict_test1(self): 
        # self.x_train.reshape(-1, self.dim_x)[:, 0]
        N_mesh = 100
        x_test = np.tile(np.linspace(-0.5, 1.5, N_mesh).reshape(-1, 1), (1, self.dim_x))
        # x_test = np.tile(np.linspace(0, 1, N_mesh).reshape(-1, 1), (1, self.dim_x))
        X1, X2 = np.meshgrid(x_test[:, 0], x_test[:, 1])
        X_TEST = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=-1)

        N_inputs, _ = X_TEST.shape
        N_ensemble = self.N_ensemble

        # ddd
        y_predict = self.model.predict(X_TEST)

        yy = np.zeros([N_ensemble, N_inputs, self.dim_y])


        # fig, ax = plt.subplots()
        yp = np.zeros([N_ensemble, N_inputs])
        for m in range(N_ensemble):
            yy[m, :, :] = y_predict[m]

        # Y = yy[1, :, 0].reshape(N_mesh, N_mesh)

        fig = plt.figure(figsize=(8,6))
        ax3d = plt.axes(projection="3d")

        ax3d = plt.axes(projection='3d')

        yy = np.minimum(1, yy)
        yy = np.maximum(-1, yy)
        # for m in range(N_ensemble): 
        for m in range(1): 
            ax3d.plot_surface(X1, X2, yy[m, :, 0].reshape(N_mesh, N_mesh),cmap='plasma')
        ax3d.set_title('Surface Plot in Matplotlib')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_zlim([-1, 1])

        plt.show()


plt.show()



if __name__ == "__main__":
    
    run = RUN_PREDICT()
    run.run()