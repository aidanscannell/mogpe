import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

color_1 = 'olive'
color_2 = 'darkmagenta'
color_2 = 'darkslategrey'
color_3 = 'darkred'
color_3 = 'lime'
color_obs = 'red'


class Plotter:
    def __init__(self, model, X, Y, test_input, colors=None):
        self.model = model
        self.X = X
        self.Y = Y
        self.test_input = test_input
        self.f_means, self.f_vars = model.predict_experts_fs(test_input)
        self.prob_a_0, _ = model.predict_mixing_probs(test_input)
        self.h_mu, self.h_var = model.predict_gating_h(test_input)
        self.y_means, self.y_vars = model.predict_experts_ys(test_input)
        self.y_mean, self.y_var = model.predict_y_moment_matched(test_input)
        if colors is None:
            self.colors = [color_1, color_2]
        else:
            self.colors = colors
        self.linestyles = ['-.', 'dashed']

        params = {
            'axes.labelsize': 15,
            'font.size': 15,
            'text.usetex': True,
            'figure.figsize': [10, 5.1]
        }
        plt.rcParams.update(params)

    def init_subplot_1(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5.1))
        params = {
            # 'axes.labelsize': 30,
            # 'font.size': 30,
            # 'legend.fontsize': 20,
            # 'xtick.labelsize': 30,
            # 'ytick.labelsize': 30,
            'text.usetex': True,
        }
        plt.rcParams.update(params)
        fig.tight_layout()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('$\mathbf{x}$')
        ax.set_ylabel('$\mathbf{y}$')
        return fig, ax

    def init_subplot_21(self):
        fig, axs = plt.subplots(nrows=1,
                                ncols=2,
                                figsize=(17, 5.1),
                                dpi=100,
                                sharex='all')
        # sharey='row')
        params = {
            'axes.labelsize': 30,
            'font.size': 30,
            'legend.fontsize': 20,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'text.usetex': True,
        }
        axs[0].set_ylim(-3, 3)
        axs[1].set_ylim(-3, 3)
        plt.rcParams.update(params)
        fig.tight_layout()
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('$\mathbf{x}$')
            ax.set_ylabel('$\mathbf{y}$')
        return fig, axs

    def init_subplot_12(self):
        fig, axs = plt.subplots(nrows=2,
                                ncols=1,
                                figsize=(17, 10.1),
                                sharex='all')
        # sharey='row')
        params = {
            'axes.labelsize': 30,
            'font.size': 30,
            'legend.fontsize': 20,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'text.usetex': True,
        }
        axs[0].set_ylim(0, 1)
        axs[1].set_ylim(-3, 3)
        plt.rcParams.update(params)
        fig.tight_layout()
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        axs[1].set_xlabel('$\mathbf{x}$')
        axs[0].get_xaxis().set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        return fig, axs

    def init_subplot_22(self):
        fig, axs = plt.subplots(nrows=2, ncols=2, sharey='row', sharex='all')
        for ax in axs.flatten():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('$\mathbf{x}$')
            for spine in ax.spines.values():
                spine.set_position(('outward', 5))
        for ax in axs[0, :].flatten():
            ax.get_xaxis().set_visible(False)
            ax.spines['bottom'].set_visible(False)
        return fig, axs

    def load_svgp_data(self, filename='./logs/svgp/y_samples.npz'):
        svgp = np.load('./logs/svgp/y_samples.npz')
        self.input_samples = svgp['input_samples']
        self.input_broadcast = svgp['input_broadcast']
        self.y_mean_svgp = svgp['y_mean']
        self.y_var_svgp = svgp['y_var']
        self.y_samples_svgp = svgp['y_samples']

    def plot_observations(self, fig, ax, color=color_obs, size=10):
        ax.scatter(self.X,
                   self.Y,
                   marker='x',
                   color=color,
                   lw=0.4,
                   s=size,
                   alpha=0.8,
                   label='Observations')

    def plot_mean(self, fig, ax, input, mean, color='k'):
        ax.plot(input,
                mean,
                color=color,
                linestyle="-",
                lw=0.6,
                label='Predictive mean')

    def plot_variance(self, fig, ax, input, mean, var, color='k'):
        ax.fill_between(
            input[:, 0],
            mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
            mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
            color=color,
            alpha=0.2,
        )

    def plot_gp(self, fig, ax, input, mean, var, color="C0", linestyle='-'):
        self.plot_mean(fig, ax, input, mean, color=color)
        self.plot_variance(fig, ax, input, mean, var, color=color)

    def plot_samples(self, fig, ax, input_broadcast, y_samples, color=color_3):
        ax.scatter(input_broadcast,
                   y_samples,
                   marker='.',
                   s=4.9,
                   color=color,
                   lw=0.4,
                   rasterized=True,
                   alpha=0.2)

    def plot_y_samples(self,
                       fig,
                       ax,
                       input_samples,
                       input_broadcast,
                       y_mean,
                       y_var,
                       y_samples,
                       size=10):
        self.plot_mean(fig, ax, input_samples, y_mean, color='k')
        self.plot_samples(fig, ax, input_broadcast, y_samples)
        self.plot_observations(fig, ax, size=size)
        # self.plot_variance(fig, ax, input_samples, y_mean, y_var, color='k')
        ax.legend()

    def plot_samples_both(self):
        fig, axs = self.init_subplot_21()
        self.load_svgp_data(filename='./logs/svgp/y_samples.npz')
        y_samples = self.model.sample_y(self.input_samples)
        y_mean, y_var = self.model.predict_y_moment_matched(self.input_samples)
        self.plot_y_samples(fig, axs[0], self.input_samples,
                            self.input_broadcast, y_mean, y_var, y_samples)
        self.plot_y_samples(fig, axs[1], self.input_samples,
                            self.input_broadcast, self.y_mean_svgp,
                            self.y_var_svgp, self.y_samples_svgp)

    def plot_prob(self, fig, ax, prob, color, k, label_k=False):
        if label_k:
            label = "k=" + str(k)
        else:
            label = '$P(\\alpha_*=' + str(
                k) + ' | \mathbf{x}_*, \mathcal{D}, \\phi)$'
        ax.plot(self.test_input, prob, color=color, label=label)
        ax.set_ylim([0, 1])
        ax.set_ylabel('$P(\\alpha=' + str(k) + ' | \mathbf{x})$')

    def plot_experts_and_gating(self):
        fig, axs = self.init_subplot_22()
        self.plot_prob(fig, axs[0, 0], self.prob_a_0, color_1, k=1)
        self.plot_prob(fig, axs[0, 1], 1 - self.prob_a_0, color_2, k=2)
        self.plot_observations(fig, axs[1, 0])
        self.plot_observations(fig, axs[1, 1])
        axs[1, 0].set_ylabel('$\mathbf{y}^{(1)}$')
        axs[1, 1].set_ylabel('$\mathbf{y}^{(2)}$')
        self.plot_gp(fig, axs[1, 0], self.test_input, self.y_means[0],
                     self.y_vars[0], self.colors[0], self.linestyles[0])
        self.plot_gp(fig, axs[1, 1], self.test_input, self.y_means[1],
                     self.y_vars[1], self.colors[1], self.linestyles[1])

    def plot_y_svmogpe(self, fig=None, ax=None, size=10):
        if fig is None:
            fig, ax = self.init_subplot_1()
        self.load_svgp_data(filename='./logs/svgp/y_samples.npz')
        # y_samples = self.model.sample_y(self.input_samples)
        y_samples = self.model.sample_y(self.test_input)
        # self.plot_samples(fig, ax, self.input_broadcast, y_samples)
        input_broadcast = np.tile(self.test_input, (100, 1, 1))
        ax.plot(self.test_input,
                self.y_means[0],
                color=self.colors[0],
                linestyle=self.linestyles[0],
                lw=0.6,
                label='Predictive mean expert 1')
        ax.plot(self.test_input,
                self.y_means[1],
                color=self.colors[1],
                linestyle=self.linestyles[1],
                lw=0.6,
                label='Predictive mean expert 2')
        self.plot_y_samples(fig,
                            ax,
                            self.test_input,
                            input_broadcast,
                            self.y_mean,
                            self.y_var,
                            y_samples,
                            size=size)
        ax.legend()

    def plot_y_svmogpe_and_gating_network(self, fig, axs):
        params = {
            'axes.labelsize': 30,
            'font.size': 30,
            'legend.fontsize': 17,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'text.usetex': True,
        }
        plt.rcParams.update(params)
        self.plot_y_svmogpe(fig, axs[0])
        axs[0].legend(loc=3)
        self.plot_prob(fig, axs[1], self.prob_a_0, color_1, k=1, label_k=True)
        self.plot_prob(fig,
                       axs[1],
                       1 - self.prob_a_0,
                       color_2,
                       k=2,
                       label_k=True)
        label = '$\Pr(\\alpha_*=k | \mathbf{x}_*, \mathcal{D}, \\phi)$'
        # label = ""
        axs[1].set_ylabel(label)
        yticks = np.linspace(0, 1, 3)
        axs[1].set_yticks(yticks)
        axs[1].legend(loc=3)

    # def plot_y_svmogpe_and_gating_side(self):
    #     fig, axs = self.init_subplot_21()
    #     params = {
    #         'axes.labelsize': 30,
    #         'font.size': 30,
    #         'legend.fontsize': 17,
    #         # 'legend.fontsize': 12,
    #         'xtick.labelsize': 30,
    #         'ytick.labelsize': 30,
    #         'text.usetex': True,
    #     }
    #     plt.rcParams.update(params)
    #     self.plot_prob(fig, axs[1], self.prob_a_0, color_1, k=1, label_k=True)
    #     self.plot_prob(fig,
    #                    axs[1],
    #                    1 - self.prob_a_0,
    #                    color_2,
    #                    k=2,
    #                    label_k=True)
    #     label = '$\Pr(\\alpha_*=k | \mathbf{x}_*, \mathcal{D}, \\phi)$'
    #     # label = ""
    #     axs[1].set_ylabel(label)
    #     self.load_svgp_data(filename='./logs/svgp/y_samples.npz')
    #     y_samples = self.model.sample_y(self.test_input)
    #     input_broadcast = np.tile(self.test_input, (100, 1, 1))
    #     axs[0].scatter(self.X,
    #                    self.Y,
    #                    marker='x',
    #                    color=color_obs,
    #                    lw=0.4,
    #                    s=10,
    #                    alpha=0.8)
    #     self.plot_samples(fig, axs[0], input_broadcast, y_samples)
    #     self.plot_mean(fig, axs[0], self.test_input, self.y_mean, color='k')
    #     # self.plot_y_samples(fig,
    #     #                     axs[1],
    #     #                     self.test_input,
    #     #                     input_broadcast,
    #     #                     self.y_mean,
    #     #                     self.y_var,
    #     #                     y_samples,
    #     #                     size=10)
    #     axs[0].plot(self.test_input,
    #                 self.y_means[0],
    #                 color=self.colors[0],
    #                 linestyle=self.linestyles[0],
    #                 lw=0.9,
    #                 label='Predictive mean expert 1')
    #     axs[0].plot(self.test_input,
    #                 self.y_means[1],
    #                 color=self.colors[1],
    #                 linestyle=self.linestyles[1],
    #                 lw=0.9,
    #                 label='Predictive mean expert 2')
    #     yticks = np.linspace(0, 1, 3)
    #     print(yticks)
    #     axs[1].set_yticks(yticks)
    #     axs[0].legend(loc=3)
    #     axs[1].legend(loc=3)


# if __name__ == "__main__":
#     from util import init_svmogpe, load_mcycle_dataset, load_mixture_dataset
#     dataset_name = 'sin'
#     # dataset_name = 'mcycle'
#     if dataset_name == 'sin':
#         (X, Y), _, _ = load_mixture_dataset(
#             filename=
#             '../data/artificial/artificial-1d-mixture-sin-gating-sin-expert-higher-noise.npz',
#             standardise=False)
#         num_test = 400
#         x_min = X.numpy().min() * 2
#         x_max = X.numpy().max() * 2
#         # prior_name = 'expert_rbf_gating_rbf'
#         # save_dir = './saved_model_rbf_gating_kernel_rbf_expert'
#         # prior_name = 'expert_rbf_gating_sin'
#         # save_dir = './saved_model_composite_gating_kernel_rbf_epxert'
#         prior_name = 'expert_sin_gating_rbf'
#         save_dir = './saved_model_rbf_gating_kernel'
#         prior_name = 'expert_sin_gating_sin'
#         save_dir = './saved_model_composite_gating_kernel'
#     else:
#         X, Y = load_mcycle_dataset(filename='~/Developer/datasets/mcycle.csv')
#         num_test = 200
#         x_min = X.numpy().min()
#         x_max = X.numpy().max()
#         save_dir = './saved_model_svmogpe_mcycle'
#     # x_min = X.numpy().min() * 1.2
#     # x_max = X.numpy().max() * 1.2
#     # x_min = 0
#     # x_max = 100
#     # x_min = X.numpy().min() * 3
#     # x_max = X.numpy().max() * 3
#     input = np.linspace(x_min, x_max, num_test).reshape(-1, 1)

#     # save_dir = './logs/svmogpe/mcycle/tight/analytic-f/05-13-144031/saved_model'

#     # save_dir = './saved_model_composite_gating_kernel'
#     # save_dir = './logs/svmogpe/mcycle/tight/sample-f/events.out.tfevents.1588612191.dr-robots-mbp.local.15796.474.v2'

#     loaded_model = tf.saved_model.load(save_dir)
#     # loaded_result_2 = loaded_model.sample_y(input, num_samples=num_samples_y)

#     plotter = Plotter(loaded_model, X, Y, input)

#     if dataset_name == 'mcycle':
#         plotter.plot_mcycle_svgp_comparison_and_gating()
#         plt.savefig("../images/model/" + dataset_name + "/svgp_comparison.pdf",
#                     transparent=True)
#         plotter.plot_mcycle_svgp_comparison()
#         plt.savefig("../images/model/" + dataset_name +
#                     "/svgp_comparison_no_gating.pdf",
#                     transparent=True)
#         # plotter.plot_y_svmogpe_and_gating()
#         # plt.savefig("../images/model/" + dataset_name + "/y_svmgpe_and_gating.pdf",
#         #             transparent=True)
#         plotter.plot_mcycle_svgp_comparison_var()
#         plt.savefig("../images/model/" + dataset_name +
#                     "/svgp_comparison_var.pdf",
#                     transparent=True)
#         plt.show()

#     # plotter.plot_ys()
#     else:
#         plotter.plot_experts_and_gating()
#         plt.savefig("../images/model/" + dataset_name + "/" + prior_name +
#                     "/experts_and_gating" + str(num_test) + ".pdf",
#                     transparent=True)
#         # plotter.plot_y_both()
#         # plt.savefig("../images/model/y_both.pdf", transparent=True)
#         plotter.plot_y_svmogpe()
#         plt.savefig("../images/model/" + dataset_name + "/" + prior_name +
#                     "/y_svmogpe" + str(num_test) + ".pdf",
#                     transparent=True)

#         # plotter.plot_samples_both()
#         # plt.savefig("../images/model/" + dataset_name + "/" + prior_name +
#         #             "/y_svgp_comparison" + str(num_test) + ".pdf",
#         #             transparent=True)

#         plotter.plot_y_svmogpe_and_gating()
#         plt.savefig("../images/model/" + dataset_name + "/" + prior_name +
#                     "/y_svmgpe_and_gating" + str(num_test) + ".pdf",
#                     transparent=True)
#         plotter.plot_y_svmogpe_and_gating_side()
#         plt.savefig("../images/model/" + dataset_name + "/" + prior_name +
#                     "/y_svmgpe_and_gating" + str(num_test) + "_side.pdf",
#                     transparent=True)

#         plt.show()
