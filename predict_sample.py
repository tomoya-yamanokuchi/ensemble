import json
import os
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from config_sample import reload_config, get_image_config
from dnn_sample import DNNSample



path_conf = "/home/dl-box/jst/python_code/ensemble/logs/N_ensemble5_20201030145502"

config = get_image_config()
config.FLAGS.reload_model = path_conf + "/"
config = reload_config(config.FLAGS)

dnn = DNNSample(config)
model = tf.keras.models.load_model( path_conf + "/model.h5", 
                                    custom_objects={'swish': dnn.swish })

# model.summary()

# ==============
#     train 
# ==============
N_train = 2000
x_train1 = np.linspace(-np.pi*2.0, -np.pi, int(N_train*0.5))
x_train2 = np.linspace( np.pi,  np.pi*2.0, int(N_train*0.5))
x_train  = np.hstack([x_train1, x_train2])

y_mean  = np.sin(x_train)
y_train = y_mean + np.random.randn(N_train) * np.std(4*0.225*np.abs(np.sin(1.5 * x_train + np.pi/8.0)))

fig, ax = plt.subplots(1,1, figsize=(8, 5))
ax.plot(x_train, y_train, "x", color="g")
# plt.show()

# ==============
#     test
# ==============
x_test = np.linspace(-np.pi*5.0, np.pi*5.0, 2000)
y_test = np.sin(x_test)

ax.plot(x_test, y_test, color="k")
# plt.show()


y_predict = model.predict(x_test)

N_ensemble = len(y_predict)
for n in range(N_ensemble): 
    ax.plot(x_test, y_predict[n], color="r")
plt.show()
