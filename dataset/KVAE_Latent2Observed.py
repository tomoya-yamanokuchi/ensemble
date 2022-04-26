from .DatasetAbstract import DatasetAbstract
import numpy as np


class KVAE_Latent2Observed(DatasetAbstract):
    def load_train(self):
        path = "/hdd_mount/logs/dclaw_64x64_N2001_seq25_dim_a8_Epoch2000_seed0_SCREW_TELLO_20220422181628_random_nonfix_to_canonical_kvae/test_generation/run_KRCAN_latent_get_z_state_Epochs_1999_20220426221006"
        x    = np.load(path + "/z_filtered_latent.npy").astype(np.float32)
        y    = np.load(path + "/z_true.npy").astype(np.float32)
        return x, y


    def load_test(self, N=2000):
        path = "/hdd_mount/logs/dclaw_64x64_N2001_seq25_dim_a8_Epoch2000_seed0_SCREW_TELLO_20220422181628_random_nonfix_to_canonical_kvae/test_generation/run_KRCAN_latent_get_z_state_Epochs_1999_20220426221006"
        x    = np.load(path + "/z_filtered_latent.npy").astype(np.float32)
        y    = np.load(path + "/z_true.npy").astype(np.float32)
        return x, y

