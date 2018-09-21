"""
Exercise: Regularization

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
    mean = np.mean(d)
    sigma = np.std(d)

    return  (d - mean) / sigma


def matricization(x):
    X = np.vstack([
        np.ones(x.shape[0]),
        x,
        np.power(x, 2),
        np.power(x, 3),
        np.power(x, 4),
        np.power(x, 5),
        np.power(x, 6),
        np.power(x, 7),
        np.power(x, 8),
        np.power(x, 9),
        np.power(x, 10)
        ])

    return X.T


def f_w(x, w):
    return np.dot(x, w)


def g(x):
    return 0.1 * (x + np.power(x, 2) + np.power(x, 3))


def error(x, y):
    return 0.5 + np.sum((y - x) ** 2)


def regression(x, y, w, rate_learning, is_regular=True):
    counter_iter = 1
    error_diff = sys.maxsize
    error_prev = sys.maxsize
    error_stop = 0.00001
    max_iteration = 50000
    LAMBDA = 10

    # DO REGRESSION
    while error_diff > error_stop:
        if is_regular:
            # L1-Norm
            regular = np.hstack([0, w[1:]])
            w = w - rate_learning * (np.dot(f_w(x, w) - y, x) + (LAMBDA * regular))
        else:
            w = w - rate_learning * np.dot(f_w(x, w) - y, x)

        # update error_diff
        error_current = error(f_w(x, w), y)
        error_diff = error_prev - error_current
        error_prev = copy.copy(error_current)

        # stop by max_iteration
        if counter_iter < max_iteration:
            pass
        else:
            break

        print(counter_iter, w, error_current, error_diff, "+r" if is_regular else "-r")
        counter_iter += 1

    print()

    return w


def draw_plot(x, y, w1, w2):
    # scatter-plot
    fig, ax = plt.subplots(figsize=(PLOT_X_SIZE, PLOT_Y_SIZE), facecolor='w')

    # draw plot
    z = np.array(np.linspace(-2.1, 2.1, 100))
    z_m = matricization(normalize_data(z))
    plt.plot(x, y, "o", color="b", ms=7, mew=1)
    plt.plot(z, g(z), color="r", ls="--")
    plt.plot(z, f_w(z_m, w1), color="g", ls="--")
    plt.plot(z, f_w(z_m, w2), color="b", ls="--")

    # plot setting
    ax.grid(color="k", linestyle="dotted", linewidth=0.8, alpha=0.8)
    ax.set_xlabel(r"$x$", fontdict={"fontsize": 12})
    ax.set_ylabel(r"$f_\theta(x)$", fontdict={"fontsize": 12})
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.0, 3.0)
    ax.set_title("Regularization", fontdict={"fontsize": 12})
    ax.tick_params(axis="both", direction="in", which="major", labelsize=8)

    # legend-text
    legend_text = ["noised data", "true-curve", "-regular", "+regular"]
    ax.legend(legend_text, loc=4, fontsize="small", prop={"size": 8}, ncol=1, framealpha=1)

    # save image
    plt.tight_layout()
    plt.savefig("regularization.png", dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close()


# --------------------------------------------------------------------------------
# 4.Main function
# --------------------------------------------------------------------------------
def main_function():
    # input, output
    x = np.array(np.linspace(-2, 2, 8))
    y = g(x) + (np.random.rand(len(x)) * 0.05)  # g(x) + noise
    X = matricization(normalize_data(x))

    # --------------------------------------------------
    # logistic regression with regularization
    rate_learning = 0.0001
    w1 = np.random.rand(X.shape[1])
    w2 = np.random.rand(X.shape[1])
    w1 = regression(X, y, w1, rate_learning, is_regular=False)
    w2 = regression(X, y, w2, rate_learning, is_regular=True)

    # --------------------------------------------------
    # draw scatter-line plot and contour-plot
    draw_plot(x, y, w1, w2)


if __name__ == '__main__':
    main_function()
