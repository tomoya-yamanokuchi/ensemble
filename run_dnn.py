import json
import os
import time
from scipy import stats
import subprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick 
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from mpl_toolkits.mplot3d import Axes3D

from dnn_kvae import DNNModel
from config_kvae import reload_config, get_image_config
from myCallBack import MYCallBack


class RUN_DNN:
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


    def plot_inputs_scatter(self, x, y): 
        cm = plt.cm.get_cmap('RdYlBu')

        fig = plt.figure()        
        ax = fig.add_subplot(1, 1, 1)
        
        for i in range(x.shape[0]): 
            print(" - {0}/{1}".format(i, x.shape[0]))
            mappable = ax.scatter(x[i, :, 0], x[i, :, 1], c=y[i, :, 0], vmin=0, vmax=20, s=10, cmap=cm, lw=0)

        fig.colorbar(mappable, ax=ax)
        plt.show()


        # import seaborn as sns
        # import pandas as pd
        # import matplotlib.cm as cm

        # data = pd.DataFrame(x, columns=["state", "control"])
        # ax1 = sns.jointplot(x="state", y='control', data=data, kind='hex')
        # # ax1.ax_joint.cla()
        # # plt.sca(ax1.ax_joint)
        # # plt.hist2d(data["state"], data["control"], bins=(100, 100), cmap=cm.jet)

        plt.show()


    def run(self): 
        config = get_image_config()
        config = reload_config(config.FLAGS)
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
        # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        # Save hyperparameters
        config.log_dir = "logs"
        run_name = "N_ensemble{}_".format(config.N_ensemble) + time.strftime('%Y%m%d%H%M%S', time.localtime())
        config.log_dir = os.path.join(config.log_dir, run_name)
        if not os.path.isdir(config.log_dir):
            os.makedirs(config.log_dir)
        with open(config.log_dir + '/config.json', 'w') as f:
            json.dump(config.flag_values_dict(), f)


        # =================
        #     training
        # =================
        npzfile  = np.load(config.dataset)
        y_train  = npzfile['pred_error'].astype(np.float32)
        x_train1 = npzfile['z'].astype(np.float32)
        x_train2 = npzfile['u'].astype(np.float32)
        x_train3 = npzfile['alpha'].astype(np.float32)
        x_train  = np.concatenate([x_train1, x_train2, x_train3], axis=-1)

        N_train, step, dim_x = x_train.shape
        dim_y = config.dim_outputs


        # self.plot_inputs_scatter(x_train, np.log(y_train))
        # self.plot_inputs_scatter(x_train.reshape(-1, dim_x)[:, :2], np.log(y_train).reshape(-1, dim_y))

        # self.plot_hist(x_train1.reshape(-1))
        # self.plot_hist(np.log(x_train1.reshape(-1)))
        # self.plot_hist(x_train1.reshape(-1))

        # self.plot_hist(x_train3.reshape(-1))

        # for i in range(dim_x): 
        #     self.plot_line(x_train[:, :, i])

        x_max = np.max(np.max(x_train, axis=0), axis=0).reshape(1, 1, dim_x)
        x_min = np.min(np.min(x_train, axis=0), axis=0).reshape(1, 1, dim_x)
        x_train = (x_train - x_min) / (x_max - x_min)


        # for i in range(dim_x): 
        #     self.plot_line(x_train[:, :, i])

        # self.plot_hist(x_train.reshape(-1, 2)[:, 0])
        # self.plot_hist(x_train.reshape(-1, 2)[:, 1])
        # self.plot_hist(x_train1.reshape(-1))

        # plt.figure()
        # sns.heatmap(x_train.reshape(-1, 2))
        # plt.show()

        # fig, ax = plt.subplots(1,2, figsize=(12, 5))
        # for i in range(N_train):
        #     for d in range(2): 
        #         ax[d].plot(x_train[i, :, d])
        # plt.show()


        # y_max = np.max(np.max(y_train, axis=0), axis=0).reshape(1, 1, dim_y)
        # y_min = np.min(np.min(y_train, axis=0), axis=0).reshape(1, 1, dim_y)
        # y_train = (y_train - y_min) / (y_max - y_min)
        # y_train = y_train * config.scale_inputs

        y_train = np.log(y_train + config.bias)
        y_train = (y_train - np.mean(y_train.reshape(-1))) / np.std(y_train.reshape(-1))

        self.plot_hist(y_train.reshape(-1))

        # fig, ax = plt.subplots()
        # for i in range(N_train):
        #     ax.plot(y_train[i, :, 0])
        # # ax.set_yscale('log')
        # plt.show()


        # # ================
        # #   3D plot error
        # # ================
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # cm = plt.cm.get_cmap('hsv')
        # mappable = ax.scatter(x_train.reshape(-1, 2)[:, 0], x_train.reshape(-1, 2)[:, 1], y_train.reshape(-1), c=y_train.reshape(-1), cmap=cm)
        # ax.set_xlabel("state")
        # ax.set_ylabel("control")
        # ax.set_zlabel("prediction error")
        # fig.colorbar(mappable, ax=ax)
        # plt.show()

        # ================
        #   histgram
        # ================
        g = y_train.reshape(-1)
        fig, ax = plt.subplots()
        plt.hist(g)
        plt.show()



        # =================
        #    validation 
        # =================
        N_valid = int(N_train*0.3)
        ind_valid = range(N_valid)
        ind_valid = np.random.choice(ind_valid, N_valid, replace=True)
        y_valid = y_train[ind_valid]
        x_valid = x_train[ind_valid]

        # fig, ax = plt.subplots(1,2, figsize=(12, 5))
        # for i in range(N_valid):
        #     for d in range(2): 
        #         ax[d].plot(x_valid[i, :, d])
        # plt.show()


        # x_test = x_train.reshape(-1, 9, 2)
        # N_test = x_test.shape[0]
        # fig, ax = plt.subplots()
        # for i in range(N_test):
        #     ax.plot(x_test[i, :, 0])
        # plt.show()


        # # for i in range(N_train): 
        # #     ax.hist(y_train[i])
        # ax.hist(y_train)
        # plt.show()

        # fig, ax = plt.subplots()
        # for i in range(N_train): 
        #     ax.plot(x_train1[i, :, 0])
        # plt.show()

        # fig, ax = plt.subplots()
        # for i in range(N_train): 
        #     ax.plot(x_train2[i, :, 0])
        # plt.show()


        session_config = ConfigProto()
        session_config.gpu_options.allow_growth = True

        dnn = DNNModel(config)
        # model = dnn.nn_constructor()

        model = dnn.nn_ensemble(N_ensemble=config.N_ensemble)
        model.summary()

        # example_batch = x_test
        # example_result = model.predict(example_batch)
        # print(example_result)


        # optimizer = tf.keras.optimizers.Adam(0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)

        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        # model.compile(loss='msle', optimizer=optimizer, metrics=['msle'])


        checkpoint_path = config.log_dir + "/cp-{epoch:04d}.ckpt"
        checkpoint_dir  = os.path.dirname(checkpoint_path)
        cp_callback     = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                        save_weights_only=True,
                                                        verbose=0,
                                                        period=config.epoch)

        mycb = MYCallBack()


        session_config = ConfigProto()
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            model.fit(x_train.reshape(-1, dim_x), [y_train[:,:,0].reshape(-1)]*config.N_ensemble, 
                                epochs=config.epoch,
                                batch_size=config.batch_size,  
                                validation_data=(x_valid.reshape(-1, dim_x), [y_valid[:,:,0].reshape(-1)]*config.N_ensemble), 
                                # callbacks=[cp_callback, mycb], 
                                callbacks=[mycb], 
                                use_multiprocessing=True)

            model.save(checkpoint_dir + "/model.h5")


        print("end")



if __name__ == "__main__":
    
    run = RUN_DNN()
    run.run()
