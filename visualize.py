import matplotlib.pyplot as plt
import numpy as np
import torch

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def plot_training_curves(nll_log, constant_log, distance_log, output_filename, silent=True):
    """ Plot training curve and dist + constant progression """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    x_plt = list(nll_log.keys())
    y_plt = list(nll_log.values())
    ax[0].set_title('Loss Curve')
    ax[0].set_ylabel('$-log(p(z|\mu, p))$')
    ax[0].set_xlabel('Epoch')
    ax[0].plot(x_plt, y_plt)
    ax[0].ticklabel_format(useOffset=False)

    distances_plot = list(distance_log.values())
    constants_plot = list(constant_log.values())
    ax[1].set_title('Distance vs Constant plot')
    ax[1].plot(x_plt, distances_plot, label="distance")
    ax2 = ax[1].twinx()
    ax2.plot(x_plt, constants_plot, c="red", label="constant")
    ax[1].legend(loc="best")
    ax2.legend(loc="best")

    fig.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.show()
    if silent:
        plt.close()


def plot_mu_curve(mu_log, output_filename, silent=True):
    """ mu progression """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    x_plt = list(mu_log.keys())
    mu = np.stack(list(mu_log.values()))
    ax[0].set_title('$\mu_1$-Change')
    ax[0].set_ylabel('$\mu_1$ values')
    ax[0].set_xlabel('Epoch')
    ax[0].plot(x_plt, mu[:, 0], label="$\mu_1$")

    ax[1].set_title('$\mu_2$-Change')
    ax[1].set_ylabel('$\mu_2$ values')
    ax[1].set_xlabel('Epoch')
    ax[1].plot(x_plt, mu[:, 1], label="$\mu_2$")
    fig.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.show()
    if silent:
        plt.close()


def plot_training_curves(nll_log, test_nll_log, output_filename, silent=True):
    """ Plot training curve and dist + constant progression """
    fig, ax = plt.subplots(figsize=(7, 7))

    x_plt = list(nll_log.keys())
    y_train_plt = list(nll_log.values())
    y_test_plt = list(test_nll_log.values())
    ax.set_title('Loss Curve')
    ax.set_ylabel('$-log(p(z|\mu, p))$')
    ax.set_xlabel('Epoch')
    ax.plot(x_plt, y_train_plt, label='Training loss')
    ax.plot(x_plt, y_test_plt, label='Test loss')
    ax.ticklabel_format(useOffset=False)
    ax.legend(loc="best")

    fig.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.show()
    if silent:
        plt.close()


def plot_covariance(std_log, output_filename, silent=True):
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    std_stack = torch.stack(list(std_log.values()))
    cov_stack = torch.inverse(torch.transpose(std_stack, 1, 2).bmm(std_stack))
    x_plt = list(std_log.keys())

    for idx, sub_ax in enumerate(ax.flat):
        sigma = '$\sigma_' + str(idx + 1) + '$'
        sub_ax.set_title(sigma + '-Change')
        sub_ax.set_ylabel(sigma + ' values')
        sub_ax.set_xlabel('Epoch')

    ax[0, 0].plot(x_plt, cov_stack[:, 0, 0], label="$\sigma_1$")
    ax[0, 1].plot(x_plt, cov_stack[:, 0, 1], label="$\sigma_2$")
    ax[1, 0].plot(x_plt, cov_stack[:, 1, 0], label="$\sigma_3$")
    ax[1, 1].plot(x_plt, cov_stack[:, 1, 1], label="$\sigma_4$")
    plt.suptitle('Covariance matrix plot', y=1.05, fontsize=16)
    fig.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.show()
    if silent:
        plt.close()


def plot_std(std_log, output_filename, silent=True):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    std_stack = np.stack(list(std_log.values()))
    x_plt = list(std_log.keys())

    ax.set_title('$\sigma$-Change')
    ax.set_ylabel('$\sigma$ values')
    ax.set_xlabel('Epoch')
    ax.plot(x_plt, std_stack, label="$\sigma_1$")

    fig.tight_layout()
    plt.savefig(output_filename)
    if silent:
        plt.close()


def plot_eigenvalues(std_log, output_filename, silent=True):
    n_epochs = len(std_log.keys())
    test = torch.cat(list(std_log.values())).reshape(n_epochs, 2, 2)
    eig_log = {}
    for i in range(test.shape[0]):
        eig_values = (test[i].T @ test[i]).inverse().eig()
        eig_log[i] = eig_values[0][:, 0].sort(descending=True)[0]

    eig_values = torch.cat(list(eig_log.values())).reshape(n_epochs, 2).numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    x_plt = list(std_log.keys())

    ax[0].plot(x_plt, eig_values[:, 0])
    ax[0].set_xlabel('Epoch')
    ax[0].set_title('Eigen value 1')
    ax[0].ticklabel_format(useOffset=False)

    ax[1].plot(x_plt, eig_values[:, 1])
    ax[1].set_xlabel('Epoch')
    ax[1].set_title('Eigen value 2')
    ax[1].ticklabel_format(useOffset=False)

    fig.tight_layout()
    plt.suptitle('Eigen values plot', y=1.05, fontsize=16)
    plt.savefig(output_filename, bbox_inches="tight")
    plt.show()
    if silent:
        plt.close()
