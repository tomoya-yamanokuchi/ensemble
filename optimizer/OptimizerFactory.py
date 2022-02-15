from .AdamOptimizer import AdamOptimizer


class OptimizerFactory:
    def create(self, optimizer: str, config):
        assert type(optimizer) == str

        optimizer = optimizer.lower()

        dataset_dict = {
            "adam" : AdamOptimizer().get(config),
        }

        return dataset_dict[optimizer]


if __name__ == '__main__':
    factory = OptimizerFactory()

