import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras.layers import Lambda


class DNNModel: 
    def __init__(self, config):
        self.config = config
        self.units = [int(f) for f in config.units.split(',')]
        self.dim_inputs  = config.dim_inputs
        self.dim_outputs = config.dim_outputs
        print("init !")

    def swish(self, x):
        return x * keras.backend.sigmoid(x)


    def loss_gauss(self, predicted_y, target_y):
        print(predicted_y.shape)
        return tf.reduce_mean(tf.square(predicted_y - target_y))


    def nn_construct(self):
        inputs  = keras.layers.Input(shape=(self.dim_inputs,))
        x       = Dense(self.units[0], activation=self.swish)(inputs)

        for n_unit in self.units[1:]: 
            x = Dense(n_unit, activation=self.swish)(x)

        outputs = Dense(self.dim_outputs)(x)
        return keras.Model(inputs, outputs)


    def nn_ensemble(self, N_ensemble):
        self.N_ensemble = N_ensemble
        model = []
        for n in range(N_ensemble): 
            model.append(self.nn_construct())

        inputs = keras.Input(shape=(self.dim_inputs,))

        y = []
        for n in range(N_ensemble): 
            y.append(model[n](inputs))

        # outputs = layers.average(y)
        outputs = y
        ensemble_model = keras.Model(inputs=inputs, outputs=outputs)

        return ensemble_model
