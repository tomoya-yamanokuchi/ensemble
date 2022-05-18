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
from loss import myloss
import PlotHandler as plothandler

from dnn_kvae import DNNModel
# from config_kvae import reload_config, get_image_config
from config_seesaw import reload_config, get_image_config
from myCallBack import MYCallBack
from Repository import Repository
import ConsoleOutput  as cout
import Normalization as norm

from dataset.DatasetFactory import DatasetFactory
from dnn_state_estimator import DNNStateEstimator


class RunDNNSingleNetwork:
    def __init__(self, config):
        self.config     = config
        factory         = DatasetFactory()
        self.dataset    = factory.create(dataset_name=config.dataset, config=config)
        self.repository = Repository()
        self.repository.save_config(self.config)

    def run(self):
        config = copy.deepcopy(self.config)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

        x_train, y_train     = self.dataset.load_train()
        N_train, step, dim_x = x_train.shape
        dim_y                = y_train.shape[-1]
        y_train_origin       = copy.deepcopy(y_train)

        cout.console_output(N_train, step, dim_x, dim_y)

        # -------------- normalize and convert -------------------
        # x_train, x_min , x_max = norm.normalize(x_train)
        # y_train                = np.log(y_train)
        # y_train, y_min , y_max = norm.normalize(y_train)
        # repository.save_norm_data(config.log_dir + "/norm_data.npz",  x_min, x_max, y_min, y_max)

        y_train, y_log_mean, y_log_std = norm.normalize_z_score(y_train)
        self.repository.save_norm_data_z_score(config.log_dir + "/norm_data.npz",  y_log_mean, y_log_std)

        # plothandler.plot_all_sequence(x_train[:, :, :])
        # plothandler.plot_all_sequence(y_train[:, :, :])

        # -------------- split data -------------------
        _, x_valid = train_test_split(x_train, test_size=0.3)
        _, y_valid = train_test_split(y_train, test_size=0.3)

        # -------------- split data -------------------
        checkpoint_path = config.log_dir + "/cp-{epoch:04d}.ckpt"
        checkpoint_dir  = os.path.dirname(checkpoint_path)
        cp_callback     = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=0,
                                                        period=config.epoch)

        mycb = MYCallBack(config.log_dir)


        name_list = ["dense_" + str(n) for n in range(len(config.units.split(",")) + 1)]

        session_config = tf.compat.v1.ConfigProto()
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            with tf.variable_scope(config.scope_name):
                dnn       = DNNStateEstimator(config)
                model     = dnn.nn_construct()

            # 変数名は読み込む時には正しいとは限らないので保存しておく
            variable_name_list = [variable.name for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=config.scope_name)]
            self.repository.save_variable_name_list(variable_name_list, save_path=config.log_dir)

            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            model.compile(loss=myloss.smooth_L1, optimizer=optimizer, metrics=['mse'])
            # model.compile(loss='msle', optimizer=optimizer, metrics=['msle'])
            model.summary()

            saver = tf.train.Saver(model.trainable_weights)
            sess.run(tf.initialize_all_variables())
            model.fit(
                x                   = x_train.reshape(-1, dim_x),
                y                   = y_train.reshape(-1, dim_y),
                epochs              = config.epoch,
                batch_size          = config.batch_size,
                validation_data     = (x_valid.reshape(-1, dim_x), y_valid.reshape(-1, dim_y)),
                callbacks           = [mycb],
                use_multiprocessing = True
            )
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

    # @hydra.main(config_path="conf/config_ICRA2022.yaml")
    # @hydra.main(config_path="conf/config_test.yaml")
    @hydra.main(config_path="conf/config_RAL_revise.yaml")
    def get_config(cfg: DictConfig) -> None:

        run = RunDNNSingleNetwork()
        run.run(cfg)

    get_config()


