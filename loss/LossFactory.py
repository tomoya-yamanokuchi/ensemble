from .myloss import *


class LossFactory:
    def create(self, loss):
        assert type(loss) == str

        loss = loss.lower()

        dataset_dict = {
            "smooth_l1" : smooth_L1,
            "mse"       : "mean_squared_error",
            "mae"       : "mean_absolute_error",
        }

        return dataset_dict[loss]


if __name__ == '__main__':
    LossFactory().create(loss="smooth_L1")
    LossFactory().create(loss="mse")
    LossFactory().create(loss="mae")

