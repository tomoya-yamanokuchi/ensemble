from cProfile import label
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np

def plot_line(x, y, is_reurn_figax=False):
    fig, ax = plt.subplots(1,1)
    ax.plot(x, y)
    plt.show()
    if   is_reurn_figax : return fig, ax
    else                : plt.cla()


def plot_marker(x, y, is_reurn_figax=False):
    fig, ax = plt.subplots(1,1)
    ax.plot(x, y, linestyle="", marker="x")
    plt.show()
    if   is_reurn_figax : return fig, ax
    else                : plt.cla()


def plot_ensemble_result(x_train, y_train, x_test, y_predict: list, y_true, figsize, ylim, save_dir):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.plot(x_test, y_true, color="#90A4AE", label="true")
    ax.plot(x_train, y_train, linestyle="", marker="x", color="#2E7D32", alpha=0.6, label="training")
    # ------------------------------------------------
    color_list = ["#00BFA5", "#AA00FF", "#FD9A28", "#37DC94", "#268AFF"]
    for n in range(len(y_predict)):
        ax.plot(x_test, y_predict[n],  color=color_list[n], label="Network{}".format(n))
    # ------------------------------------------------
    if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(np.minimum(x_test.min(), x_train.min()), np.maximum(x_test.max(), x_train.max()))
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    plt.legend(loc='upper left', borderaxespad=0, fontsize=18)
    # plt.show()
    plt.savefig(save_dir + "/ensemble_result.png")



def plot_ensemble_result_mean_var(x_train, y_train, x_test, y_predict: list, y_true, figsize, ylim, save_dir):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.plot(x_test, y_true, color="#90A4AE", label="true")
    ax.plot(x_train, y_train, linestyle="", marker="x", color="#2E7D32", alpha=0.6, label="training")
    # ------------------------------------------------
    y_concat = np.concatenate(y_predict, axis=-1)
    y_mean   = np.mean(y_concat, axis=-1)
    y_std    = np.std(y_concat, axis=-1)
    ax.plot(x_test, y_mean, color="#F50057", label="prediction mean")
    ax.fill_between(x_test, (y_mean - 2.0 * y_std), (y_mean + 2.0 * y_std), color="#EC407A", alpha=0.3, label="prediction std")
    # ------------------------------------------------
    if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(np.minimum(x_test.min(), x_train.min()), np.maximum(x_test.max(), x_train.max()))
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    plt.legend(loc='upper left', borderaxespad=0, fontsize=18)
    # plt.show()
    plt.savefig(save_dir + "/ensemble_result_mean_var.png")





