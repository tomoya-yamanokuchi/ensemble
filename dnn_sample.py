import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


class DNNSample: 
    def __init__(self, config):
        self.config = config
        self.dim_input  = 1
        self.dim_output = 1
        print("init !")

    def swish(self, x):
        return x * keras.backend.sigmoid(x)

    def nn_construct(self):
        inputs  = keras.layers.Input(shape=(self.dim_input,))
        # x       = Dense(32, activation='relu')(inputs)
        # x       = Dense(128, activation='relu')(x)
        # x       = Dense(32, activation='relu')(x)
        x       = Dense(512, activation=self.swish)(inputs)
        x       = Dense(512, activation=self.swish)(x)
        x       = Dense(512, activation=self.swish)(x)
        outputs = Dense(self.dim_output)(x)
        return keras.Model(inputs, outputs)
        

    def nn_ensemble(self, N_ensemble):
        self.N_ensemble = N_ensemble
        model = []
        for n in range(N_ensemble): 
            model.append(self.nn_construct())

        inputs = keras.Input(shape=(self.dim_input,))

        y = []
        for n in range(N_ensemble): 
            y.append(model[n](inputs))

        # outputs = layers.average(y)
        outputs = y
        ensemble_model = keras.Model(inputs=inputs, outputs=outputs)

        return ensemble_model
