import json
import os
import pickle
import time
import subprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick 


class PlotLearningInfo:
    def plot(self, path): 

        with open(path + "/learning_info.pickle", "rb") as f:
            data = pickle.load(f)

        plt.plot(data["loss"])
        plt.show()



if __name__ == "__main__":
    
    run = PlotLearningInfo()
    path = "/hdd_mount/ensemble/logs/N_ensemble1_20201106195954"
    run.plot(path)