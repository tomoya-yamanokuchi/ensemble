import imp
import numpy
from .DatasetAbstract import DatasetAbstract
from tensorflow import keras
import numpy as np


class SinDataset(DatasetAbstract):
    def load_train(self, config, N=2000):
        N_train  = N
        x_train1 = np.linspace(-np.pi*2.0, -np.pi, int(N_train*0.5))
        x_train2 = np.linspace( np.pi,  np.pi*2.0, int(N_train*0.5))
        x_train  = np.hstack([x_train1, x_train2])

        y_train  = np.sin(x_train)
        y_noise  = np.random.randn(N_train) * np.sqrt(0.0225 * np.abs(np.sin(1.5 * x_train + np.pi/8.0)))
        y_train  = y_train + y_noise

        return x_train, y_train


    def load_valid(self, N=500):
        x        = np.linspace(-np.pi*2.0, np.pi*2.0, N)
        y        = np.sin(x)
        y_noise  = np.random.randn(N) * np.sqrt(0.0225 * np.abs(np.sin(1.5 * x + np.pi/8.0)))
        y        = y + y_noise
        return x, y


    def load_test(self):
        N_test = 2000
        x_test = np.linspace(-np.pi*5.0, np.pi*5.0, N_test)
        y_test = np.sin(x_test)

        return x_test, y_test

