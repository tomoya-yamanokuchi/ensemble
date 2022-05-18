
import numpy as np

import sys
import pathlib
p = pathlib.Path(__file__).resolve()
sys.path.append(str(p.parent))
repository_path = "/".join(str(p.parent).split("/")[:-2])

from DatasetAbstract import DatasetAbstract


class KVAE_Latent2Observed(DatasetAbstract):
    def __init__(self):
        self.kvae_path = None

    def load_train(self):
        assert type(self.kvae_path) == str
        path = repository_path + "/logs/{}/z_latent_dataset".format(self.kvae_path)
        x    = np.load(path + "/z_filtered_latent.npy").astype(np.float32)
        y    = np.load(path + "/z_true.npy").astype(np.float32)
        return x, y

    def load_test(self, N=2000):
        return self.load_train()

