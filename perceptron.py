"""
Exercise: Perceptron

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
ROUND_DIGITS = 8

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


def f_w(w, x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1


def perceptron(x, y, w):
    max_epoch = 25

    # DO PERCEPTRON
    for i in range(max_epoch):
        counter_w = 1

        for xi, yi in zip(x, y):
            if f_w(w, xi) != yi:
                w = w + yi * xi
            else:
                # w = w
                pass

            print(i + 1, counter_w, w, f_w(w, xi), yi)
            counter_w += 1

        print()

    return w


def draw_plot(x, y, w):
    # scatter-plot
    fig, ax = plt.subplots(figsize=(PLOT_X_SIZE, PLOT_Y_SIZE), facecolor='w')

    # draw plot
    x1_p = np.linspace(0, 525)
    x2_p = (-w[0] / w[1]) * x1_p
    plt.plot(x[y == 1, 0], x[y == 1, 1], "o", color="b", ms=7, mew=1)
    plt.plot(x[y == -1, 0], x[y == -1, 1], "x", color="r", ms=7, mew=2)
    plt.plot(x1_p, x2_p, color="r", ls="--")

    # plot setting
    ax.grid(color="k", linestyle="dotted", linewidth=0.8, alpha=0.8)
    ax.set_xlabel(r"$x_1$", fontdict={"fontsize": 12})
    ax.set_ylabel(r"$x_2$", fontdict={"fontsize": 12})
    ax.set_xlim(0, 525)
    ax.set_ylim(0, 525)
    ax.set_title("Perceptron", fontdict={"fontsize": 12})
    ax.tick_params(axis="both", direction="in", which="major", labelsize=8)

    # legend-text
    legend_text = ["class1", "class2", "perceptron"]
    ax.legend(legend_text, loc=4, fontsize="small", prop={"size": 8}, ncol=1, framealpha=1)

    # save image
    plt.tight_layout()
    plt.savefig("perceptron.png", dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close()


# --------------------------------------------------------------------------------
# 4.Main function
# --------------------------------------------------------------------------------
def main_function():
    # input, output
    d = np.loadtxt("data2.csv", delimiter=",")
    x = d[:, 0:2]
    y = d[:, 2]

    # --------------------------------------------------
    # Perceptron
    w = np.random.rand(2)
    w = perceptron(x, y, w)

    # --------------------------------------------------
    # draw scatter-line plot and contour-plot
    draw_plot(x, y, w)


if __name__ == '__main__':
    main_function()
