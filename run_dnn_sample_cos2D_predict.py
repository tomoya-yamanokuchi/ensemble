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
        x_test, y_true   = dataset.load_test(M=100)

        dnn   = DNNSample(config)
        pwd   = os.getcwd().split("outputs")[0] # This depends on hydra
        model = tf.keras.models.load_model(pwd + "logs/" + config.reload_model + "/model.h5", custom_objects={'swish': dnn.swish })

        y_predict      = model.predict(x_test)

        y_std = np.std(np.concatenate(y_predict, axis=-1), axis=-1)

        # pltsrv.plot_line(np.linspace(0, y_std.shape[0]-1, y_std.shape[0]), np.sort(y_std))
        # pltsrv.plot_line(np.linspace(0, y_std.shape[0]-1, y_std.shape[0]), np.log(np.sort(y_std)))
        # pltsrv.plot_line(np.linspace(0, y_std.shape[0]-2, y_std.shape[0]-1), np.diff(np.sort(y_std)))

        # pltsrv.plot_2D_scatter(x_train, y_train, figsize=(10, 10))
        # pltsrv.plot_2D(x_test, y_std, figsize=(10, 10))


        # vmax_list = [0.05*(i+1) for i in range(30)]
        vmax_list = [0.1]
        for i, vmax in enumerate(vmax_list):
            pltsrv.plot_2D_scatter_with_colormap(
                x=x_test, y=y_std, vlim=(0.0, vmax),
                figsize=(12, 10), s=40, xlim=(-11, 11), ylim=(-11, 11),
                save_name = "test_std_vmax_index{}_model_{}".format(i, config.reload_model),
                # save_name = "test_std_vmax_index{}".format(i),
                title_str = "vmax_{:>.2}".format(vmax),
            )




if __name__ == "__main__":
    import hydra
    from attrdict import AttrDict
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_path="conf/config_sample_cos2D.yaml")
    def get_config(cfg: DictConfig):
        run = RUN_DNN()
        run.run(cfg)

    get_config()


