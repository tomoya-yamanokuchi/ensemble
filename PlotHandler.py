import matplotlib


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd



def plot_line(x):
    seq, step = x.shape
    fig, ax = plt.subplots()
    for s in range(seq):
        ax.plot(x[s, :])
    plt.show()
    print()


def plot_hist(x):
    fig, ax = plt.subplots()
    plt.hist(x)
    print("======================")
    print('       var: {0:.2f}'.format(np.var(x)))
    print('       std:{0:.2f}'.format(np.std(x)))
    print('      skew: {0:.2f}'.format(stats.skew(x)))
    print('  kurtosis: {0:.2f}'.format(stats.kurtosis(x)))
    print("======================")
    plt.show()


def plot_all_sequence(x):
    assert len(x.shape) == 3
    sequence, step, dim = x.shape
    fig, ax = plt.subplots(dim, 1)
    for d in range(dim):
        if dim > 1:
            ax[d].plot(x[:, :, d].transpose())
        else:
            ax.plot(x[:, :, d].transpose())
    plt.show()



def plot_inputs_scatter(x, y, x_colomuns):
    # cm = plt.cm.get_cmap('RdYlBu')

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    # for i in range(x.shape[0]):
    #     print(" - {0}/{1}".format(i, x.shape[0]))
    #     mappable = ax.scatter(x[i, :, 0], x[i, :, 1], c=y[i, :, 0], vmin=0, vmax=20, s=10, cmap=cm, lw=0)

    # mappable = ax.scatter(x[i, :, 0], x[i, :, 1], c=y[i, :, 0], vmin=0, vmax=20, s=10, cmap=cm, lw=0)
    # fig.colorbar(mappable, ax=ax)
    # plt.show()

    data = pd.DataFrame(x, columns=x_colomuns)
    ax1 = sns.jointplot(x="state", y='control', data=data, kind='hex')
    # ax1.ax_joint.cla()
    # plt.sca(ax1.ax_joint)
    # plt.hist2d(data["state"], data["control"], bins=(100, 100), cmap=cm.jet)

    plt.show()



# def plot_data_info(self, x_train, y_train):
#     self.plot_hist(y_train.reshape(-1))
#     self.plot_hist(x_train[:, :, -8:].reshape(-1))
#     self.plot_hist(x_train[:, :, 6:-8].reshape(-1))
#     self.plot_all_sequence(np.log(y_train))

#     for i in range(dim_x):
#         self.plot_line(x_train[:, :, i])
#         # self.plot_hist(x_train[:, :, i].reshape(-1))


#     self.plot_hist(y_train.reshape(-1))

#     fig, ax = plt.subplots()
#     for i in range(N_train):
#         ax.plot(y_train[i, :, 0])
#     plt.show()




def predict_both(x_train, y_train, x_test, y_test, y_predict, N_test):
    sequence, step, dim_y, ensemble_num = y_predict.shape

    dy = 0

    fig, ax = plt.subplots(2, N_test, figsize=(9, 6))
    for n in range(N_test):
        for m in range(ensemble_num):
            ax[0, n].plot(y_predict[n, :, dy, m], color="mediumvioletred", label="DNN predict")
        ax[0, n].plot(y_test[n, :, dy], color="k", label="Ground Truth")

        mean  = np.mean(y_predict[n, :, dy, :], axis=-1)
        std   = np.std( y_predict[n, :, dy, :], axis=-1)
        lower = mean - 2.0*std
        upper = mean + 2.0*std
        x = range(step)
        color_fill = "thistle"
        ax[1, n].fill_between(x, lower, upper, alpha=0.6, color=color_fill)
        ax[1, n].plot(x, mean,  "-",  color="mediumvioletred", label="DNN predict")
        ax[1, n].plot(y_test[n, :, dy], color="k", label="Ground Truth")

        ax[1, n].set_xlabel("Step", fontsize=18)
        if n == 0:
            ax[0, n].set_ylabel(r"$ e_{t_+1} $", fontsize=18)
            ax[1, n].set_ylabel(r"$ e_{t_+1} $", fontsize=18)

    lines = []
    labels = []
    for _ax in fig.axes:
        axLine, axLabel = _ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    # fig.legend(lines, labels[:5], bbox_to_anchor=(0.75, 0.95,), ncol=5, fontsize=16)
    fig.legend(lines[5:7], labels[5:7], loc="upper center", ncol=2, fontsize=14)

    plt.show()