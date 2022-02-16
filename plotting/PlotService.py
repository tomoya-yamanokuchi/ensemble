from cProfile import label
from tkinter import font
from cv2 import sqrt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
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


def plot_ensemble_result(x_train, y_train, x_test, y_predict: list, y_true, figsize, ylim, save_dir, config=None):
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
    if config is None:  name = "/ensemble_result.png"
    else:               name = "/ensemble_result_dataset_{}_epoch_{}.png".format(config.dataset, config.epoch)
    plt.savefig(save_dir + name)



def plot_ensemble_result_mean_var(x_train, y_train, x_test, y_predict: list, y_true, figsize, ylim, save_dir, config=None):
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
    if config is None:  name = "/ensemble_result_mean_var.png"
    else:               name = "/ensemble_result_mean_var_dataset_{}_epoch_{}.png".format(config.dataset, config.epoch)
    plt.savefig(save_dir + name)


def plot_2D(x, y, figsize, colorbar_minmax=None):
    x1, x2 = np.split(x, x.shape[-1], axis=-1)
    N = y.shape[0]
    y = y.reshape(int(np.sqrt(N)), int(np.sqrt(N)))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if colorbar_minmax is not None:
        im = ax.imshow(y, cmap='coolwarm', norm=Normalize(vmin=colorbar_minmax[0], vmax=colorbar_minmax[1]))
    else:
        im = ax.imshow(y, cmap='coolwarm')
    plt.colorbar(im)
    ax.set_xlabel(r"x",fontsize=20)
    ax.set_ylabel(r"y",fontsize=20)
    # ax.tick_params(axis='x1', labelsize=20)
    # ax.tick_params(axis='x2', labelsize=20)
    plt.show()





def plot_3D_surface(x, y, figsize):
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    x1, x2  = np.split(x, x.shape[-1], axis=-1)
    N       = y.shape[0]
    x1_grid = x1.reshape(int(np.sqrt(N)), int(np.sqrt(N)))
    x2_grid = x2.reshape(int(np.sqrt(N)), int(np.sqrt(N)))
    y_grid  = y.reshape(int(np.sqrt(N)), int(np.sqrt(N)))

    surf = ax.plot_surface(x1_grid, x2_grid, y_grid, cmap="coolwarm", linewidth=0,
            # norm=Normalize(vmin=0.0, vmax=0.05),
            antialiased=False)

    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.show()







def plot_3D_surface_ensemble(x, y_predict: list, figsize):
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    x1, x2  = np.split(x, x.shape[-1], axis=-1)
    N       = y_predict[0].shape[0]
    x1_grid = x1.reshape(int(np.sqrt(N)), int(np.sqrt(N)))
    x2_grid = x2.reshape(int(np.sqrt(N)), int(np.sqrt(N)))

    for y in y_predict:
        y_grid  = y.reshape(int(np.sqrt(N)), int(np.sqrt(N)))
        surf = ax.plot_surface(x1_grid, x2_grid, y_grid, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()




def plot_3D_scatter(x, y, figsize):
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    x1, x2  = np.split(x, x.shape[-1], axis=-1)
    N       = y.shape[0]
    ax.scatter(x1.reshape(-1), x2.reshape(-1), y, s = 5, c = "blue")
    plt.show()
    
    
    
def plot_3D_scatter_with_true(x, y, x_true, y_true, figsize, config=None):
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    x1, x2  = np.split(x, x.shape[-1], axis=-1)
    N       = y.shape[0]
    
    x_true1, x_true2  = np.split(x_true, x_true.shape[-1], axis=-1)
    
    ax.scatter(x_true1.reshape(-1), x_true2.reshape(-1), y_true, s = 5, c = "gray")
    ax.scatter(x1.reshape(-1), x2.reshape(-1), y, s = 5, c = "blue")
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    
    plt.show()    
    save_dir = "/home/tomoya-y/Pictures"
    if config is None:  name = "/ensemble_result.png"
    else:               name = "/ensemble_result_dataset_{}_epoch_{}.png".format(config.dataset, config.epoch)
    plt.savefig(save_dir + name)



def plot_2D_scatter(x, y, figsize, s=10, xlim=None, ylim=None, save_name=None):
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111)

    x1, x2  = np.split(x, x.shape[-1], axis=-1)
    N       = y.shape[0]
    ax.scatter(x1.reshape(-1), x2.reshape(-1), s = s, c = "green")
    ax.set_xlabel("x1", fontsize=16)
    ax.set_ylabel("x2", fontsize=16)
    if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
    save_dir = "/home/tomoya-y/Pictures"
    name = "/{}.png".format(save_name)
    plt.savefig(save_dir + name)
    plt.show()
    
    
    
    
def plot_2D_scatter_with_colormap(x, y, vlim, figsize, x_train=None, s=10, xlim=None, ylim=None, save_name=None, title_str=""):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x1, x2  = np.split(x, x.shape[-1], axis=-1)
    N       = y.shape[0]
    sc = ax.scatter(x1.reshape(-1), x2.reshape(-1), vmin=vlim[0], vmax=vlim[1], s=s, c=y, cmap=cm.coolwarm)
    
    if x_train is not None: 
        x_train1, x_train2  = np.split(x_train, x_train.shape[-1], axis=-1)
        ax.scatter(x_train1.reshape(-1), x_train2.reshape(-1), s=10, c="gray", marker="x")
    
    cb = fig.colorbar(sc)
    fontsize = 20
    cb.set_label(label='ensemble std', size=fontsize)
    ax.set_xlabel("x1", fontsize=fontsize)
    ax.set_ylabel("x2", fontsize=fontsize)
    ax.set_title(title_str, fontsize=fontsize)
    if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
    save_dir = "/home/tomoya-y/Pictures"
    name = "/{}.png".format(save_name)
    plt.savefig(save_dir + name)
    # plt.show()
    




def plot_3D_scatter_ensemble(x_train, y_train, x, y_predict, figsize):
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    x_train1, x_train2  = np.split(x_train, x_train.shape[-1], axis=-1)
    ax.scatter(x_train1.reshape(-1), x_train2.reshape(-1), y_train, s = 40, c = "green", marker="x")

    x1, x2  = np.split(x, x.shape[-1], axis=-1)
    N       = y_predict[0].shape[0]
    for y in y_predict:
        ax.scatter(x1.reshape(-1), x2.reshape(-1), y, s = 40)
    plt.show()

