from .DatasetAbstract import DatasetAbstract
import numpy as np


class SincDataset(DatasetAbstract):
    def load_train(self, config, N=2000):
        x1      = np.linspace(-np.pi*2.0, -np.pi*0.1, int(N*0.5))
        x2      = np.linspace( np.pi*0.1,  np.pi*2.0, int(N*0.5))
        x       = np.hstack([x1, x2])

        y       = np.sinc(x)
        y_noise = np.random.randn(N) * np.sqrt(0.02 * np.abs(np.sinc(1.5 * x + np.pi/8.0)))
        y       = y + y_noise

        return x, y


    def load_test(self, N=2000):
        x = np.linspace(-np.pi*3.0, np.pi*3.0, N)
        y = np.sinc(x)
        return x, y

