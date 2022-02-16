from .SinDataset import SinDataset
from .SincDataset import SincDataset
from .cos2DDataset import cos2DDataset


class DatasetFactory:
    def create(self, dataset_name: str):
        assert type(dataset_name) == str

        dataset_dict = {
            "sin" : SinDataset(),
            "sinc" : SincDataset(),
            "cos2D" : cos2DDataset(),
        }

        return dataset_dict[dataset_name]


if __name__ == '__main__':
    factory = DatasetFactory()
    factory.create(dataset_name="sin")