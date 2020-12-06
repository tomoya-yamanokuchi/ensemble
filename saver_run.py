import json
import os
import pickle
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
# from config_kvae import reload_config, get_image_config
from config_seesaw import reload_config, get_image_config
from myCallBack import MYCallBack


class RUN_DNN:
    def create_data(self):
        N = 1000
        dim_x = 15
        x = np.random.random([N, dim_x])
        y = np.random.random(N)
        return x, y

    def run(self): 
        config = get_image_config()
        config = reload_config(config.FLAGS)
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

        # Save hyperparameters
        config.log_dir = "logs"
        run_name = "ensemble_M{}_".format(config.N_ensemble) + "saver_sample" + "_" + time.strftime('%Y%m%d%H%M%S', time.localtime())
        config.log_dir = os.path.join(config.log_dir, run_name)
        if not os.path.isdir(config.log_dir):
            os.makedirs(config.log_dir)
        with open(config.log_dir + '/config.json', 'w') as f:
            json.dump(config.flag_values_dict(), f, ensure_ascii=False, indent=4, separators=(',', ': '))

        x, y = self.create_data()

        session_config = ConfigProto()
        session_config.gpu_options.allow_growth = True
        with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
            with tf.name_scope("ensemble"):

                dnn = DNNModel(config)
                model = dnn.nn_ensemble(N_ensemble=config.N_ensemble)
            

                optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)

                model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

                checkpoint_path = config.log_dir + "/cp-{epoch:04d}.ckpt"
                checkpoint_dir  = os.path.dirname(checkpoint_path)
                cp_callback     = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                                save_weights_only=True,
                                                                verbose=0,
                                                                period=config.epoch)

                mycb = MYCallBack(config.log_dir)

                saver = tf.train.Saver()
                model.fit(x, [y]*config.N_ensemble, 
                            epochs=config.epoch,
                            batch_size=config.batch_size,  
                            callbacks=[mycb], 
                            use_multiprocessing=True)

                # model.save(checkpoint_dir + "/model.h5")
                saver.save(sess, checkpoint_dir + '/model.ckpt')

        print("end")



if __name__ == "__main__":
    
    run = RUN_DNN()
    run.run()
