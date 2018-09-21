"""
Exercise: Linear Regression using stochastic gradient descent (SGD)

@auth: Yu-Hsiang Fu
@date: 2018/09/18
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


def f_theta(x, theta):
    return theta[0] + (x * theta[1])


def error(x, y):
    return 0.5 * (np.sum((y - x) ** 2))


def normalize_data(d):
    mean = np.mean(d)
    sigma = np.std(d)

    return (d - mean) / sigma


def stochastic_gradient_descent(x, y, theta, rate_learning, mini_batch):
    counter_iter = 1
    error_diff = sys.maxsize
    error_prev = sys.maxsize
    error_stop = 0.0001
    max_iteration = 1000
    theta_history = [p_deepcopy(theta)]

    # DO gradient-descent
    while error_diff > error_stop:
        # decide amount (all or mini_batch) of x-data
        p = np.random.choice(x.shape[0], size=mini_batch, replace=False)

        # update theta using each x-data
        for i in p:
            theta0 = theta[0] - rate_learning * np.sum(f_theta(x[i], theta) - y[i])
            theta1 = theta[1] - rate_learning * np.sum((f_theta(x[i], theta) - y[i]) * x[i])
            theta = [theta0, theta1]

        # update error_diff
        error_current = error(f_theta(x, theta), y)
        error_diff = error_prev - error_current
        error_prev = copy.copy(error_current)

        # store updated-results
        theta_history.append(p_deepcopy(theta))

        # stop by max_iteration
        if counter_iter < max_iteration:
            pass
        else:
            break

        print(counter_iter, theta, error_current, error_diff, error_diff > error_stop)
        counter_iter += 1

    return theta_history


def draw_plot(x, y, theta_history):
    # contour-plot
    # plot-xy-range
    x_left, x_right = -100, 900
    y_left, y_right = -200, 400

    # plot-grid
    num_level = 50
    x_ls = np.linspace(x_left, x_right, num_level)
    y_ls = np.linspace(y_left, y_right, num_level)

    X_ls, Y_ls = np.meshgrid(x_ls, y_ls)
    Z_ls = np.zeros((len(x_ls), len(y_ls)))

    for i in range(len(x_ls)):
        for j in range(len(y_ls)):

            for k in range(len(x)):
                Z_ls[j][i] += pow(y[k] - (x_ls[i] + y_ls[j] * x[k]), 2)

            Z_ls[j][i] /= len(y_ls)

    # create a figure
    fig, ax = plt.subplots(figsize=(PLOT_X_SIZE, PLOT_Y_SIZE), facecolor="w")

    # draw plot
    theta0 = [theta0 for theta0, theta1, in theta_history]
    theta1 = [theta1 for theta0, theta1, in theta_history]
    ax.plot(theta0, theta1, "o-", ms=7, lw=1.2, mew=1.2, markevery=2, fillstyle="none", c="b")

    # plot setting
    ax.contourf(X_ls, Y_ls, Z_ls, num_level, alpha=0.5, cmap=plt.get_cmap("jet"))
    ax.grid(color="k", linestyle="dotted", linewidth=0.8, alpha=0.8)
    ax.set_xlabel(r"$\theta_0$", fontdict={"fontsize": 12})
    ax.set_ylabel(r"$\theta_1$", fontdict={"fontsize": 12})
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_left, y_right)
    ax.set_title("Stochastic Gradient Descent (SGD), iter={0}".format(len(theta_history)), fontdict={"fontsize": 12})
    ax.tick_params(axis="both", direction="in", which="major", labelsize=8)

    # save the figure
    plt.tight_layout()
    plt.savefig("gradient-descent_SGD.png", dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight', pad_inches=0.05)
    plt.close()


# --------------------------------------------------------------------------------
# 4.Main function
# --------------------------------------------------------------------------------
def main_function():
    # input data
    d = np.loadtxt("data.csv", delimiter=",")

    # input, output
    x = d[:, 0]
    y = d[:, 1]

    # normalize data
    x = normalize_data(x)

    # --------------------------------------------------
    # linear-regression setting (model, parameters)
    # model: f_theta = theta_0 + (theta_1 * x)
    # parameters: theta = (theta_0, theta_1)
    theta = np.random.rand(2)

    # gradient_descent, input: x, y, theta; output: theta_history
    mini_batch = int(x.shape[0] * 0.8)
    rate_learning = 0.01
    theta_history = stochastic_gradient_descent(x, y, theta, rate_learning, mini_batch)

    # --------------------------------------------------
    # draw scatter-line plot and contour-plot
    draw_plot(x, y, theta_history)


if __name__ == '__main__':
    main_function()
