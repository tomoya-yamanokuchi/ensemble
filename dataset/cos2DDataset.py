from .DatasetAbstract import DatasetAbstract
import numpy as np


class cos2DDataset(DatasetAbstract):
    # def load_train(self):
    #     N   = 2000
    #     x11 = np.linspace(-np.pi*3.0, -np.pi, np.sqrt(int(N*0.5)))
    #     x21 = np.linspace(-np.pi*3.0, -np.pi, np.sqrt(int(N*0.5)))

    #     xx1 = np.meshgrid(x11, x21)
    #     y   = np.cos(xx1) + np.cos(xx1)

    #     x       = np.hstack([x1, x2])

    #     y       = np.sinc(x)
    #     y_noise = np.random.randn(N) * np.sqrt(0.02 * np.abs(np.sinc(1.5 * x + np.pi/8.0)))
    #     y       = y + y_noise

    #     return x, y


    def load_train(self):
        a  = 3.0
        M  = 40

        x11 = np.linspace(-np.pi*3, -np.pi*1, M)
        x21 = np.linspace(-np.pi*3, -np.pi*1, M)
        x11_grid, x21_grid = np.meshgrid(x11, x21)

        x12 = np.linspace( np.pi*1,  np.pi*3, M)
        x22 = np.linspace( np.pi*1,  np.pi*3, M)
        x12_grid, x22_grid = np.meshgrid(x12, x22)

        x1 = np.stack([x11_grid.reshape(-1), x21_grid.reshape(-1)], axis=-1)
        x2 = np.stack([x12_grid.reshape(-1), x22_grid.reshape(-1)], axis=-1)
        x  = np.concatenate([x1, x2], axis=0)

        y  = np.cos(x[:, 0]) + np.cos(x[:, 1])
        y  = y + np.random.randn(y.shape[0]) * 0.2
        return x, y


    def load_test(self, M=30):
        a  = 5.0
        x1 = np.linspace(-np.pi*a, np.pi*a, M)
        x2 = np.linspace(-np.pi*a, np.pi*a, M)

        x1, x2 = np.meshgrid(x1, x2)
        y  = np.cos(x1) + np.cos(x2)

        x = np.stack([x1.reshape(-1), x2.reshape(-1)], axis=-1)
        y = y.reshape(-1)
        return x, y

