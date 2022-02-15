from .SinDataset import SinDataset


class DatasetFactory:
    def create(self, dataset_name: str):
        assert type(dataset_name) == str

        dataset_dict = {
            "sin" : SinDataset(),
        }

        return dataset_dict[dataset_name]


if __name__ == '__main__':
    factory = DatasetFactory()
    factory.create(dataset_name="sin")