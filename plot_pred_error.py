import json
import os
import pickle
import time
from scipy import stats
import subprocess
import numpy as np
import matplotlib.pyplot as plt


model = []
model.append("seesaw_64x64_N5000_seq30_cem_1direction_with_wall_wide_20201109144516_kvae")
model.append("seesaw_64x64_N5000_seq30_cem_1direction_with_wall_wide_20201109144339_kvae")
model.append("seesaw_64x64_N5000_seq30_cem_1direction_with_wall_wide_20201112020315_kvae")
model.append("seesaw_64x64_N5000_seq30_cem_1direction_with_wall_wide_20201112020341_kvae")

y = []
for _model in model: 
    npzfile = np.load("/hdd_mount/logs/" + _model + "/1step_prediction_error_data/pred_error_from_random.npz")
    y_train = npzfile['pred_error'].astype(np.float32)
    y_train = np.expand_dims(np.sum(y_train, axis=-1), axis=-1)
    y.append(y_train.reshape(-1))

N_model = len(model)
fig, ax = plt.subplots(N_model, 1)

for i in range(N_model): 
    ax[i].hist(y[i])
    ax[i].set_xlim([0, 0.4])
    ax[i].set_ylim([0, 30])
# plt.ylim([0, 1000])
plt.show() 

print()
