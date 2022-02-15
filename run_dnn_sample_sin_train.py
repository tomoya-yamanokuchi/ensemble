import imp
import json
import os
import numpy as np

from Repository import Repository
from dnn_sample import DNNSample
from myCallBack import MYCallBack
from loss.LossFactory import LossFactory
from optimizer.OptimizerFactory import OptimizerFactory
from dataset.DatasetFactory import DatasetFactory
import plotting.PlotService as pltsrv


class RUN_DNN:
    def run(self, config):
        print("Hello!")
        self.config = config
        repository  = Repository()
        repository.save_config(config)

        factory          = DatasetFactory()
        dataset          = factory.create(dataset_name=config.dataset)
        x_train, y_train = dataset.load_train()

        # pltsrv.plot_marker(x_train, y_train)

        dnn   = DNNSample(config)
        model = dnn.nn_ensemble(N_ensemble=config.N_ensemble)
        model.summary()

        optimizer = OptimizerFactory().create(optimizer=config.optimizer, config=config)
        model.compile(loss=LossFactory().create(loss=config.loss), optimizer=optimizer, metrics=['mse'])

        checkpoint_path = config.log_dir + "/cp-{epoch:04d}.ckpt"
        checkpoint_dir  = os.path.dirname(checkpoint_path)

        model.fit(
            x                   = x_train,
            y                   = [y_train]*config.N_ensemble,
            epochs              = config.epoch,
            batch_size          = config.batch_size,
            validation_data     = (x_train, [y_train]*config.N_ensemble),
            callbacks           = [MYCallBack(config.log_dir)],
            use_multiprocessing = True
        )

        model.save(checkpoint_dir + "/model.h5")

        x_test, y_true = dataset.load_test()
        y_predict      = model.predict(x_test)

        # pltsrv.plot_ensemble_result(
        #     x_train   = x_train,
        #     y_train   = y_train,
        #     x_test    = x_test,
        #     y_predict = y_predict,
        #     y_true    = y_true,
        #     figsize   = (12, 8),
        #     ylim      = (-5.0, 5.0),
        #     save_dir  = "/home/tomoya-y/Pictures"
        # )


if __name__ == "__main__":
    import hydra
    from attrdict import AttrDict
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_path="conf/config_sin_sample.yaml")
    def get_config(cfg: DictConfig):
        run = RUN_DNN()
        run.run(cfg)

    get_config()


