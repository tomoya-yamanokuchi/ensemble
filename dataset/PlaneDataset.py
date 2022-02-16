import imp
import numpy
from .DatasetAbstract import DatasetAbstract
from tensorflow import keras
import numpy as np


class PlaneDataset(DatasetAbstract):
    def load_train(self, N=2000):
        x = np.linspace(-2*pi, 2*pi, 256)
        y = np.linspace(-3*pi, 3*pi, 256)

        # 格子点を作成
        X, Y = np.meshgrid(x, y)

        return x, y



    def load_test(self):
        N_test = 2000
        x_test = np.linspace(-np.pi*5.0, np.pi*5.0, N_test)
        y_test = np.sin(x_test)

        return x_test, y_test

