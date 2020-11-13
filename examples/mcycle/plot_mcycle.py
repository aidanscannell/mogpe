import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mogpe.visualization.plotter import Plotter


def plot_mcycle_svgp_comparison_and_gating(plotter, svgp_filename):
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all')
    for ax in axs.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('$\mathbf{x}$')
    for ax in axs[0, :].flatten():
        ax.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
    plotter.plot_y_svmogpe(fig, axs[1, 1])
    plotter.load_svgp_data(svgp_filename)
    plotter.plot_y_samples(fig, axs[0, 1], plotter.input_samples,
                           plotter.input_broadcast, plotter.y_mean_svgp,
                           plotter.y_var_svgp, plotter.y_samples_svgp)
    axs[1, 1].get_legend().remove()
    axs[0, 1].get_legend().remove()
    axs[0, 1].set_xlim(plotter.test_input.min(), plotter.test_input.max())

    axs[1, 0].set_ylabel('$\mathbf{y}$')
    axs[0, 1].set_ylabel('$\mathbf{y}$')
    axs[1, 1].set_ylabel('$\mathbf{y}$')
    axs[1, 1].set_ylim(-3, 3)
    axs[0, 1].set_ylim(-3, 3)
    axs[1, 0].set_ylim(-3, 3)
    plotter.plot_gp(fig, axs[1, 0], plotter.test_input, plotter.y_means[0],
                    plotter.y_vars[0], plotter.colors[0],
                    plotter.linestyles[0])
    plotter.plot_gp(fig, axs[1, 0], plotter.test_input, plotter.y_means[1],
                    plotter.y_vars[1], plotter.colors[1],
                    plotter.linestyles[1])
    plotter.plot_observations(fig, axs[1, 0])

    plotter.plot_prob(fig,
                      axs[0, 0],
                      plotter.prob_a_0,
                      plotter.colors[0],
                      k=1,
                      label_k=True)
    plotter.plot_prob(fig,
                      axs[0, 0],
                      1 - plotter.prob_a_0,
                      plotter.colors[1],
                      k=2,
                      label_k=True)
    axs[0, 0].set_ylabel('$p(\\alpha_*=k | \mathbf{x}_*, \mathcal{D}, \phi)$')
    axs[0, 0].legend()


def plot_mcycle_svgp_comparison(plotter, svgp_filename):
    fig, axs = plotter.init_subplot_21()
    plotter.plot_y_svmogpe(fig, axs[0], size=70)
    plotter.load_svgp_data(svgp_filename)
    plotter.plot_y_samples(fig,
                           axs[1],
                           plotter.input_samples,
                           plotter.input_broadcast,
                           plotter.y_mean_svgp,
                           plotter.y_var_svgp,
                           plotter.y_samples_svgp,
                           size=70)


def plot_mcycle_comparison_to_svgp(plotter, svgp_filename):
    fig, axs = plotter.init_subplot_21()
    fig.subplots_adjust(wspace=0.35)
    params = {
        'axes.labelsize': 30,
        'font.size': 30,
        'legend.fontsize': 19,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'text.usetex': True,
    }
    plt.rcParams.update(params)
    plotter.load_svgp_data(svgp_filename)
    color_svgp = 'red'
    color = 'blue'

    def normalise_output(output):
        return (output + 2) / 4

    axs[0].set_xlim(plotter.X.numpy().min(), plotter.X.numpy().max())
    axs[0].scatter(
        plotter.X,
        plotter.Y,
        # normalise_output(self.Y),
        marker='x',
        color='red',
        lw=0.4,
        s=70,
        alpha=0.8,
        label='Observations')
    axs[0].plot(
        plotter.test_input,
        plotter.y_mean,
        # normalise_output(self.y_mean),
        linestyle='-',
        color='k',
        label='Mean ours')
    axs[0].plot(
        plotter.input_samples,
        # normalise_output(self.y_mean_svgp),
        plotter.y_mean_svgp,
        linestyle=':',
        color='blue',
        label='Mean SVGP')
    label = '$p(\\alpha_*=1 | \mathbf{x}_*, \mathcal{D})$'
    ax2 = axs[0].twinx()
    ax2.set_ylabel('$\Pr(\\alpha_*=k | \mathbf{x}_*, \mathcal{D}, \phi)$')
    ax2.plot(plotter.test_input,
             plotter.prob_a_0,
             color=plotter.colors[0],
             label='k=1')
    ax2.plot(plotter.test_input,
             1 - plotter.prob_a_0,
             color=plotter.colors[1],
             label='k=2')
    input = plotter.input_samples
    mean = plotter.y_mean_svgp
    var = plotter.y_var_svgp
    axs[1].plot(
        input[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        # normalise_output(mean[:, 0] - 1.96 * np.sqrt(var[:, 0])),
        color=color_svgp,
        alpha=0.8,
    )
    axs[1].plot(
        input[:, 0],
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        # normalise_output(mean[:, 0] + 1.96 * np.sqrt(var[:, 0])),
        color=color_svgp,
        alpha=0.8,
    )
    input_broadcast = np.tile(plotter.input_samples, (100, 1, 1))
    y_samples = plotter.model.sample_y(plotter.input_samples)
    plotter.plot_samples(
        fig,
        axs[1],
        input_broadcast,
        y_samples,
        # normalise_output(y_samples),
        color=color)
    axs[0].legend(loc=2)
    ax2.legend(loc=1)


if __name__ == "__main__":
    from mogpe.models.utils.data import load_mcycle_dataset
    # filename = '../../models/saved_model/svgp_mcycle.npz'  # npz file with result of training svgp
    filename = './models/saved_model/svgp_mcycle.npz'  # npz file with result of training svgp
    # dataset_name = 'mcycle'
    # X, Y = load_mcycle_dataset(filename='../../data/external/mcycle.csv')
    X, Y = load_mcycle_dataset(filename='./data/external/mcycle.csv')
    num_test = 200
    x_min = X.numpy().min()
    x_max = X.numpy().max()
    save_dir = './models/saved_model/mcycle'
    input = np.linspace(x_min, x_max, num_test).reshape(-1, 1)

    loaded_model = tf.saved_model.load(save_dir)

    plotter = Plotter(loaded_model, X, Y, input)

    # plotter.plot_y_svmogpe()
    # plotter.plot_mcycle_svgp_comparison_and_gating()
    plot_mcycle_svgp_comparison_and_gating(plotter, filename)
    # plt.savefig("../images/model/" + dataset_name + "/svgp_comparison.pdf",
    #             transparent=True)
    plot_mcycle_svgp_comparison(plotter, filename)
    # plt.savefig("../images/model/" + dataset_name +
    #             "/svgp_comparison_no_gating.pdf",
    #             transparent=True)
    # plotter.plot_y_svmogpe_and_gating()
    # plt.savefig("../images/model/" + dataset_name + "/y_svmgpe_and_gating.pdf",
    #             transparent=True)
    plot_mcycle_comparison_to_svgp(plotter, filename)
    # plt.savefig("../images/model/" + dataset_name +
    #             "/svgp_comparison_var.pdf",
    #             transparent=True)
    plt.show()
