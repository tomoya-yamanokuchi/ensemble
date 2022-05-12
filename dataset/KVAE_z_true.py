import sys
import pathlib
p = pathlib.Path(__file__).resolve()
sys.path.append(str(p.parent))
sys.path.append("/".join(str(p.parent).split("/")[:-1]))
repository_path = sys.path[-1]
repository_parent_path = "/".join(repository_path.split("/")[:-1])

from .DatasetAbstract import DatasetAbstract
import numpy as np



class KVAE_z_true(DatasetAbstract):
    def __init__(self, conofig):
        self.conofig = conofig

    def load_train(self):
        path = repository_parent_path + self.conofig.dataset_path
        npz = np.load(path + "/block_mating_64x64_N2005_seq25.npz")
        state = npz["state"].astype(np.float32)
        ctrl  = npz["control"].astype(np.float32)

        x_state = state.astype(np.float32)[:, :-1, :]
        x_ctrl  = ctrl.astype(np.float32) [:, :-1, :]
        x = np.concatenate((x_state, x_ctrl), axis=-1)

        y = npz["state"].astype(np.float32)[:, 1:, :]
        return x, y

    def load_test(self, N=2000):
        return self.load_train()

