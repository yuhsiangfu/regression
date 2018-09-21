"""
Exercise: Multiple Regression

@auth: Yu-Hsiang Fu
@date: 2018/09/19
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
# EPSILON = 0.00000001
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

    return (d - mean) / sigma


def matricization(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T


def f_theta(x, theta):
    return np.dot(x, theta)


def mean_square_error(x, y):
    return (1 / x.shape[0]) * np.sum((y - x) ** 2)


def gradient_descent(x, y, theta, rate_learning):
    counter_iter = 1
    error_diff = sys.maxsize
    error_prev = sys.maxsize
    error_stop = 0.0001
    max_iteration = 1000
    error_history = [error_prev]

    # DO gradient-descent
    while error_diff > error_stop:
        # calculate gradient
        fx = f_theta(x, theta)

        # update theta
        theta = theta - rate_learning * np.dot((fx - y), x)

        # update error_diff
        error_current = mean_square_error(f_theta(x, theta), y)
        error_diff = error_prev - error_current
        error_prev = copy.copy(error_current)
        error_history.append(error_diff)

        # stop by max_iteration
        if counter_iter < max_iteration:
            pass
        else:
            break

        print(counter_iter, theta, error_current, error_diff, error_diff > error_stop)
        counter_iter += 1

    return theta, error_history


def draw_plot(x, y, theta, error_history):
    # line-plot
    fig, ax = plt.subplots(figsize=(PLOT_X_SIZE, PLOT_Y_SIZE), facecolor='w')

    # draw plot
    x_ls =np.linspace(-3.0, 3.0, 100)
    x_ls = matricization(x_ls)
    plt.plot(x[:,1], y, "o")
    plt.plot(x_ls[:,1], f_theta(x_ls, theta), color="r", ls="--")

    # plot setting
    ax.grid(color="k", linestyle="dotted", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("$x$", fontdict={"fontsize": 12})
    ax.set_ylabel(r"$f_\theta(x)$", fontdict={"fontsize": 12})
    ax.set_ylim(100, 900)
    ax.set_title("Multiple Regression", fontdict={"fontsize": 12})
    ax.tick_params(axis="both", direction="in", which="major", labelsize=8)

    # legend-text
    legend_text = ["data", "regression"]
    ax.legend(legend_text, loc=4, fontsize="small", prop={"size": 8}, ncol=1, framealpha=1)

    # save image
    plt.tight_layout()
    plt.savefig("multiple-regression.png", dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close()

    # --------------------------------------------------
    # mean-square-error line-plot
    fig, ax = plt.subplots(figsize=(PLOT_X_SIZE, PLOT_Y_SIZE), facecolor='w')

    # draw plot
    x_ls = list(range(0, len(error_history)))
    plt.plot(x_ls, error_history, color="r", ls="-")

    # plot setting
    ax.grid(color="k", linestyle="dotted", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Iteration", fontdict={"fontsize": 12})
    ax.set_ylabel("Diff. Error", fontdict={"fontsize": 12})
    ax.set_xlim(0, len(error_history))
    ax.set_ylim(-100, 10000)
    ax.set_title("Mean Square Error (MSE), iter={0}".format(len(error_history)), fontdict={"fontsize": 12})
    ax.tick_params(axis="both", direction="in", which="major", labelsize=8)

    # legend-text
    legend_text = ["MSE"]
    ax.legend(legend_text, loc=1, fontsize="small", prop={"size": 8}, ncol=1, framealpha=1)

    # save image
    plt.tight_layout()
    plt.savefig("multiple-regression_MSE.png", dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close()


# --------------------------------------------------------------------------------
# 4.Main function
# --------------------------------------------------------------------------------
def main_function():
    # data -> input x, output y
    d = np.loadtxt("data.csv", delimiter=",")
    x = d[:, 0]
    y = d[:, 1]

    # --------------------------------------------------
    # linear-regression setting (model, parameters)
    # model: f_theta = theta_0 + (theta_1 * x) + (theta_2 * x^2)
    # parameters: theta = (theta_0, theta_1, theta_2)
    # theta, matrix(x-data)
    theta = np.random.rand(3)
    x = normalize_data(x)
    X = matricization(x)

    # gradient_descent, input: x, y, theta; output: theta_history
    rate_learning = 0.01
    theta, error_history = gradient_descent(X, y, theta, rate_learning)

    # --------------------------------------------------
    # draw error-line plot
    draw_plot(X, y, theta, error_history)


if __name__ == '__main__':
    main_function()
