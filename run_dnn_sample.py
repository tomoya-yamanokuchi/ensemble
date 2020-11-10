import json
import os
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto

# from dnn_sample import DNNSample
from dnn_kvae import DNNModel
from config_sample import reload_config, get_image_config
from myCallBack import MYCallBack


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
#     train 
# ==============
N_valid = 500
x_valid1 = np.linspace(-np.pi*2.0, -np.pi, int(N_valid*0.5))
x_valid2 = np.linspace( np.pi,  np.pi*2.0, int(N_valid*0.5))
x_valid  = np.hstack([x_valid1, x_valid2])

y_mean  = np.sin(x_valid)
y_valid = y_mean + np.random.randn(N_valid) * np.std(4*0.225*np.abs(np.sin(1.5 * x_valid + np.pi/8.0)))

ax.plot(x_valid, y_valid, "x", color="m")
# plt.show()

# ==============
#     test
# ==============
x_test = np.linspace(-np.pi*5.0, np.pi*5.0, 2000)
y_test = np.sin(x_test)

ax.plot(x_test, y_test, color="k")
# plt.show()



config = get_image_config()
config = reload_config(config.FLAGS)
# os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"



# Save hyperparameters
config.log_dir = "logs"
run_name = "N_ensemble{}_".format(config.N_ensemble) + time.strftime('%Y%m%d%H%M%S', time.localtime())
config.log_dir = os.path.join(config.log_dir, run_name)
if not os.path.isdir(config.log_dir):
    os.makedirs(config.log_dir)
with open(config.log_dir + '/config.json', 'w') as f:
    json.dump(config.flag_values_dict(), f, ensure_ascii=False, indent=4, separators=(',', ': '))


session_config = ConfigProto()
session_config.gpu_options.allow_growth = True


dnn = DNNModel(config)
# model = dnn.nn_constructor()

model = dnn.nn_ensemble(N_ensemble=config.N_ensemble)
model.summary()

# example_batch = x_test
# example_result = model.predict(example_batch)
# print(example_result)


# optimizer = tf.keras.optimizers.Adam(0.001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

# model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
model.compile(loss=dnn.loss_gauss, optimizer=optimizer, metrics=['mae', 'mse'])


checkpoint_path = config.log_dir + "/cp-{epoch:04d}.ckpt"
checkpoint_dir  = os.path.dirname(checkpoint_path)
cp_callback     = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                save_weights_only=True,
                                                verbose=0,
                                                period=100)

mycb = MYCallBack()

model.fit(x_train, [y_train]*config.N_ensemble, 
                    epochs=50,
                    batch_size=64,  
                    validation_data=(x_valid, [y_valid]*config.N_ensemble), 
                    callbacks=[cp_callback, mycb], 
                    use_multiprocessing=True)

model.save(checkpoint_dir + "/model.h5")

y_predict = model.predict(x_test)

for n in range(config.N_ensemble): 
    ax.plot(x_test, y_predict[n], color="r")
plt.show()

# for n in range(config.N_ensemble): 
#     ax.plot(x_test, y_predict[n], color="r")
# plt.show()

print("end")
