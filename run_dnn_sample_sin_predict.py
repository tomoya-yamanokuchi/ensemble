import imp
import json
import os
import numpy as np
import tensorflow as tf
from Repository import Repository
from dnn_sample import DNNSample
from myCallBack import MYCallBack

from dataset.DatasetFactory import DatasetFactory
import plotting.PlotService as pltsrv


from tensorflow import keras


class RUN_DNN:
    def run(self, config):
        repository       = Repository()
        config           = repository.reload_config(config)

        factory          = DatasetFactory()
        dataset          = factory.create(dataset_name=config.dataset)
        x_train, y_train = dataset.load_train()
        x_test, y_true   = dataset.load_test()

        dnn   = DNNSample(config)
        pwd   = os.getcwd().split("outputs")[0] # This depends on hydra
        model = tf.keras.models.load_model(pwd + "logs/" + config.reload_model + "/model.h5", custom_objects={'swish': dnn.swish })

        y_predict      = model.predict(x_test)


        pltsrv.plot_ensemble_result(
            x_train   = x_train,
            y_train   = y_train,
            x_test    = x_test,
            y_predict = y_predict,
            y_true    = y_true,
            figsize   = (12, 8),
            ylim      = (-5.0, 5.0),
            save_dir  = "/home/tomoya-y/Pictures"
        )


        pltsrv.plot_ensemble_result_mean_var(
            x_train   = x_train,
            y_train   = y_train,
            x_test    = x_test,
            y_predict = y_predict,
            y_true    = y_true,
            figsize   = (12, 8),
            ylim      = (-5.0, 5.0),
            save_dir  = "/home/tomoya-y/Pictures"
        )


if __name__ == "__main__":
    import hydra
    from attrdict import AttrDict
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_path="conf/config_sin_sample.yaml")
    def get_config(cfg: DictConfig):
        run = RUN_DNN()
        run.run(cfg)

    get_config()


