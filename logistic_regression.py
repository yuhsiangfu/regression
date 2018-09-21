"""
Exercise: Logistic Regression

@auth: Yu-Hsiang Fu
@date: 2018/09/20
"""
# --------------------------------------------------------------------------------
# 1.Import packages
# --------------------------------------------------------------------------------
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys


# --------------------------------------------------------------------------------
# 2.Const variables
# --------------------------------------------------------------------------------
# program variable
# ROUND_DIGITS = 8

# plot variable
PLOT_X_SIZE = 5
PLOT_Y_SIZE = 5
PLOT_DPI = 300
PLOT_FORMAT = "png"


# --------------------------------------------------------------------------------
# 3.Define function
# --------------------------------------------------------------------------------
def p_deepcopy(data, dumps=pickle.dumps, loads=pickle.loads):
    return loads(dumps(data, -1))


def normalize_data(d):
    for i in range(d.shape[1]):
        mean = np.mean(d[:, i])
        sigma = np.std(d[:, i])
        d[:, i] = (d[:, i] - mean) / sigma

    return  d


def matricization(x):
    x0 = np.ones([x.shape[0], 1])

    return np.hstack([x0, x])


def f_w(x, w):
    return 1 / (1 + np.exp(-np.dot(x, w)))


def classify(x, w):
    return (f_w(x, w) >= 0.5).astype(np.int)


def logistic_regression(x, y, w, rate_learning):
    max_epoch = 1000

    # DO UPDATE-WEIGHT
    for i in range(max_epoch):
        w = w - rate_learning * np.dot((f_w(x, w) - y), x)

        print(i + 1, w)

    return w


def draw_plot(x, y, w):
    # scatter-plot
    fig, ax = plt.subplots(figsize=(PLOT_X_SIZE, PLOT_Y_SIZE), facecolor='w')

    # draw plot
    x1_p = np.linspace(-2, 2)
    x2_p = -((w[0] + w[1] * x1_p) / w[2])
    plt.plot(x[y == 1, 0], x[y == 1, 1], "o", color="b", ms=7, mew=1)
    plt.plot(x[y == 0, 0], x[y == 0, 1], "x", color="r", ms=7, mew=2)
    plt.plot(x1_p, x2_p, color="r", ls="--")

    # plot setting
    ax.grid(color="k", linestyle="dotted", linewidth=0.8, alpha=0.8)
    ax.set_xlabel(r"$x_1$", fontdict={"fontsize": 12})
    ax.set_ylabel(r"$x_2$", fontdict={"fontsize": 12})
    # ax.set_xlim(0, 525)
    # ax.set_ylim(0, 525)
    ax.set_title("Logistic Regression", fontdict={"fontsize": 12})
    ax.tick_params(axis="both", direction="in", which="major", labelsize=8)

    # legend-text
    legend_text = ["class1", "class2", "logistic"]
    ax.legend(legend_text, loc=4, fontsize="small", prop={"size": 8}, ncol=1, framealpha=1)

    # save image
    plt.tight_layout()
    plt.savefig("logistic-regression.png", dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close()


# --------------------------------------------------------------------------------
# 4.Main function
# --------------------------------------------------------------------------------
def main_function():
    # input, output
    d = np.loadtxt("data3.csv", delimiter=",")
    x = d[:, 0:2]
    y = d[:, 2]
    x = normalize_data(x)
    X = matricization(x)

    # --------------------------------------------------
    # logistic regression
    rate_learning = 0.01
    w = np.random.rand(3)
    w = logistic_regression(X, y, w, rate_learning)

    # --------------------------------------------------
    # draw scatter-line plot and contour-plot
    draw_plot(x, y, w)

    # --------------------------------------------------
    # # test-data
    # xx = np.matrix([[200, 100], [100, 200]])
    # xx = matricization(normalize_data(xx))
    # print(classify(xx, w))


if __name__ == '__main__':
    main_function()
