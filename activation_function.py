from tensorflow import keras

def swish(x):
    return x * keras.backend.sigmoid(x)