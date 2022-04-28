import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras.layers import Lambda
import pprint

class DNNStateEstimator:
    def __init__(self, config):
        self.config = config
        self.units = [int(f) for f in config.units.split(',')]
        self.dim_inputs  = config.dim_inputs
        self.dim_outputs = config.dim_outputs
        self.variable_name_list   = ["dense_" + str(n) for n in range(len(config.units.split(",")) + 1)]
        pprint.pprint(self.variable_name_list)


    def swish(self, x):
        return x * keras.backend.sigmoid(x)


    def loss_gauss(self, predicted_y, target_y):
        print(predicted_y.shape)
        return tf.reduce_mean(tf.square(predicted_y - target_y))


    def nn_construct(self):
        inputs  = keras.layers.Input(shape=(self.dim_inputs,))
        x = Dense(self.units[0], activation=self.swish, name=self.variable_name_list[0])(inputs)
        for i in range(len(self.units))[1:]:
            # print(i)
            x = Dense(self.units[i], activation=self.swish, name=self.variable_name_list[i])(x)
        outputs = Dense(self.dim_outputs, name=self.variable_name_list[-1])(x)
        return keras.Model(inputs, outputs)
