"""
Exercise: Logistic Regression2

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
    x3 = x[:, 0, np.newaxis] ** 2

    return np.hstack([x0, x, x3])


def f_w(x, w):
    return 1 / (1 + np.exp(-np.dot(x, w)))


def classify(x, w):
    return (f_w(x, w) >= 0.5).astype(np.int)


def logistic_regression(x, y, w, rate_learning):
    max_epoch = 1000
    accuracy_list = list()

    # DO UPDATE-WEIGHT
    for i in range(max_epoch):
        p = np.random.permutation(x.shape[0])

        for xi, yi in zip(x[p, :], y[p]):
            w = w - rate_learning * np.dot((f_w(xi, w) - yi), xi)

        classify_result = (classify(x, w) == y)
        classify_accuracy = len(classify_result[classify_result == True]) / len(y)
        accuracy_list.append(classify_accuracy)
        print(i + 1, w, round(classify_accuracy, 2))

    return w, accuracy_list


def draw_plot(x, y, w, accuracy_list):
    # scatter-plot
    fig, ax = plt.subplots(figsize=(PLOT_X_SIZE, PLOT_Y_SIZE), facecolor='w')

    # draw plot
    x1_p = np.linspace(-2.2, 2)
    x2_p = -((w[0] + (w[1] * x1_p) + (w[3] * np.power(x1_p, 2))) / w[2])
    plt.plot(x[y == 1, 0], x[y == 1, 1], "o", color="b", ms=7, mew=1)
    plt.plot(x[y == 0, 0], x[y == 0, 1], "x", color="r", ms=7, mew=2)
    plt.plot(x1_p, x2_p, color="r", ls="--")

    # plot setting
    ax.grid(color="k", linestyle="dotted", linewidth=0.8, alpha=0.8)
    ax.set_xlabel(r"$x_1$", fontdict={"fontsize": 12})
    ax.set_ylabel(r"$x_2$", fontdict={"fontsize": 12})
    ax.set_xlim(-2.2, 2.0)
    ax.set_ylim(-2.2, 1.5)
    ax.set_title("Logistic Regression", fontdict={"fontsize": 12})
    ax.tick_params(axis="both", direction="in", which="major", labelsize=8)

    # legend-text
    legend_text = ["class1", "class2", "logistic"]
    ax.legend(legend_text, loc=4, fontsize="small", prop={"size": 8}, ncol=1, framealpha=1)

    # save image
    plt.tight_layout()
    plt.savefig("logistic-regression2.png", dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close()

    # --------------------------------------------------
    # accuracy-line plot
    fig, ax = plt.subplots(figsize=(PLOT_X_SIZE, PLOT_Y_SIZE), facecolor='w')

    # draw plot
    x0_p = np.arange(len(accuracy_list))
    plt.plot(x0_p, accuracy_list, color="r", ls="-", lw=2)

    # plot setting
    ax.grid(color="k", linestyle="dotted", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Iteration", fontdict={"fontsize": 12})
    ax.set_ylabel("Accuracy", fontdict={"fontsize": 12})
    ax.set_xlim(-5, len(accuracy_list))
    ax.set_ylim(0, 1.1)
    ax.set_title("Logistic Regression", fontdict={"fontsize": 12})
    ax.tick_params(axis="both", direction="in", which="major", labelsize=8)

    # legend-text
    legend_text = ["classification accuracy"]
    ax.legend(legend_text, loc=4, fontsize="small", prop={"size": 8}, ncol=1, framealpha=1)

    # save image
    plt.tight_layout()
    plt.savefig("logistic-regression2_accuracy.png", dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close()


# --------------------------------------------------------------------------------
# 4.Main function
# --------------------------------------------------------------------------------
def main_function():
    # input, output
    d = np.loadtxt("data4.csv", delimiter=",")
    x = d[:, 0:2]
    y = d[:, 2]
    x = normalize_data(x)
    X = matricization(x)

    # --------------------------------------------------
    # logistic regression
    rate_learning = 0.01
    w = np.random.rand(4)
    w, accuracy_list = logistic_regression(X, y, w, rate_learning)

    # --------------------------------------------------
    # draw scatter-line plot and contour-plot
    draw_plot(x, y, w, accuracy_list)


if __name__ == '__main__':
    main_function()
