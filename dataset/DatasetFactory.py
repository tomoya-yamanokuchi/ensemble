from .SinDataset import SinDataset
from .SincDataset import SincDataset
from .cos2DDataset import cos2DDataset
from .KVAE_Latent2Observed import KVAE_Latent2Observed
from .KVAE_z_true import KVAE_z_true

class DatasetFactory:
    def create(self, dataset_name: str, config):
        assert type(dataset_name) == str

        dataset_dict = {
            "sin"  : SinDataset(),
            "sinc" : SincDataset(),
            "cos2D": cos2DDataset(),

            "kvae"       : KVAE_Latent2Observed(), # 状態推定器学習用
            "kvae_z_true": KVAE_z_true(config),    # アンサンブル学習用
        }

        return dataset_dict[dataset_name]


if __name__ == '__main__':
    factory = DatasetFactory()
    # factory.create(dataset_name="sin")
    factory.create(dataset_name="kvae_z_true")