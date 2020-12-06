import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras.layers import Lambda



def smooth_L1(x, y): 
    beta     = 1.0
    diff     = x - y
    diff_abs = K.abs(diff)
    mask = K.cast(K.less(diff_abs, beta), dtype='float32')

    loss = (0.5*K.square(diff) / beta) * mask - (diff_abs - 0.5*beta)*(mask-1.0)
    return loss


def smooth_L1_with_SuperLoss(x, y): 
    beta     = 1.0
    diff     = x - y
    diff_abs = K.abs(diff)
    mask = K.cast(K.less(diff_abs, beta), dtype='float32')

    loss = (0.5*K.square(diff) / beta) * mask - (diff_abs - 0.5*beta)*(mask-1.0)

    alpha = 0.25
    tau   = 0.5
    beta  = (loss - tau) / alpha
    z     = 0.5*K.maximum(-2.0/K.exp(1.0), beta)
    W     = z * K.exp(z)
    sigma = K.exp(-W)

    loss = (loss - tau)*sigma + alpha*K.square(K.log(sigma))
    return loss


def Lambert_W_function(z):
    L = K.log(z)
    M = K.log(L)
    W = M + M/L + M*(M-1)/(2*K.square(L)) + M*(6 - 9*M + 2*K.square(M))/(6*(L**3))
    return W
