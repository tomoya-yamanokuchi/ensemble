from .DatasetAbstract import DatasetAbstract
import numpy as np


class KVAE_Latent2Observed(DatasetAbstract):
    def __init__(self):
        self.kvae_path = None

    def load_train(self):
        assert type(self.kvae_path) == str
        path = "/hdd_mount/logs/{}/z_latent_dataset".format(self.kvae_path)
        x    = np.load(path + "/z_filtered_latent.npy").astype(np.float32)
        y    = np.load(path + "/z_true.npy").astype(np.float32)
        return x, y

    def load_test(self, N=2000):
        return self.load_train()

