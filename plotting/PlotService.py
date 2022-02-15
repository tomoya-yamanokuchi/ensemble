import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


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


def plot_ensemble_result(x_train, y_train, x_test, y_predict: list, y_true):
    fig, ax = plt.subplots(1,1)
    ax.plot(x_test, y_true, color="#90A4AE")
    ax.plot(x_train, y_train, linestyle="", marker="x", color="#AB47BC", alpha=0.6)
    for n in range(len(y_predict)):
        ax.plot(x_test, y_predict[n], color="#0288D1")
    plt.show()