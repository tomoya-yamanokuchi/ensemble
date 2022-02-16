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
        x_test,  y_true  = dataset.load_test(M=100)

        # pltsrv.plot_marker(x_train, y_train)
        # pltsrv.plot_3D_surface(x=x_train, y=y_train, figsize=(10, 10))
        pltsrv.plot_2D_scatter(x=x_train, y=y_train, figsize=(10, 10), s=7, xlim=(-11, 11), ylim=(-11, 11), save_name="training_data")
        pltsrv.plot_3D_scatter_with_true(x_train, y_train, x_test, y_true, figsize=(10, 10))


        dnn   = DNNSample(config)
        model = dnn.nn_ensemble(N_ensemble=config.N_ensemble)

        optimizer = OptimizerFactory().create(optimizer=config.optimizer, config=config)
        model.compile(loss=LossFactory().create(loss=config.loss), optimizer=optimizer, metrics=['mse'])
        model.summary()

        checkpoint_path = config.log_dir + "/cp-{epoch:04d}.ckpt"
        checkpoint_dir  = os.path.dirname(checkpoint_path)

        x_valid,  y_valid  = dataset.load_test(M=30)

        model.fit(
            x                   = x_train,
            y                   = [y_train]*config.N_ensemble,
            epochs              = config.epoch,
            batch_size          = config.batch_size,
            validation_data     = (x_valid, [y_valid]*config.N_ensemble),
            callbacks           = [MYCallBack(config.log_dir)],
            use_multiprocessing = True
        )

        model.save(checkpoint_dir + "/model.h5")


        y_predict      = model.predict(x_test)

        # pltsrv.plot_3D_surface_ensemble(x_test, y_predict, figsize=(10, 10))
        # pltsrv.plot_3D_scatter_ensemble(x_train, y_train, x_test, y_predict, figsize=(10, 10))




if __name__ == "__main__":
    import hydra
    from attrdict import AttrDict
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_path="conf/config_sample_cos2D.yaml")
    def get_config(cfg: DictConfig):
        run = RUN_DNN()
        run.run(cfg)

    get_config()


