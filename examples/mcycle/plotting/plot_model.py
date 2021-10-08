#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
import palettable
import tensorflow as tf
from config import config_from_toml
from mcycle.data.load_data import load_mcycle_dataset
from mogpe.training import load_model_from_config_and_checkpoint
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plt.style.use("science")
plt.style.use("seaborn-paper")
# plt.style.use("ggplot")


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

expert_colors = ["r", "g", "b"]


def init_axis_labels_and_ticks(axs):
    if isinstance(axs, np.ndarray):
        for ax in axs.flat:
            ax.set(xlabel="$x$", ylabel="$y$")
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
    else:
        axs.set(xlabel="$x$", ylabel="$y$")
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        axs.label_outer()
    return axs


class McyclePlotter:
    def __init__(
        self,
        model,
        X,
        Y,
        test_inputs=None,
        # num_samples=100,
        num_samples=50,
        # params=None,
        # num_levels=6,
        # figsize=(6.4, 4.8),
        figsize=(6.4 / 2, 4.8 / 2),
        cmap=palettable.scientific.sequential.Bilbao_15.mpl_colormap,
    ):
        self.model = model
        self.X = X
        self.Y = Y
        self.num_experts = self.model.num_experts
        self.output_dim = Y.shape[1]
        self.figsize = figsize
        self.cmap = cmap
        self.num_samples = num_samples
        if test_inputs is None:
            num_test = 100
            factor = 1.2
            self.test_inputs = tf.reshape(
                np.linspace(
                    tf.reduce_min(self.X) * factor,
                    tf.reduce_max(self.X) * factor,
                    num_test,
                ),
                [num_test, tf.shape(X)[1]],
            )
        else:
            self.test_inputs = test_inputs

        self.y_samples_dist = self.model.predict_y(self.test_inputs)
        self.y_mean = self.model.predict_y(self.test_inputs).mean()
        self.y_samples = self.y_samples_dist.sample(self.num_samples)
        self.test_inputs_broadcast = np.expand_dims(self.test_inputs, 0)
        self.svgp_mean, self.svgp_var = self.model.experts.predict_ys(self.test_inputs)
        self.svgp_mean = self.svgp_mean[:, :, 1]
        self.svgp_var = self.svgp_var[:, :, 1]

        self.y_means, self.y_vars = self.model.experts.predict_ys(self.test_inputs)
        self.f_means, self.f_vars = self.model.predict_experts_fs(self.test_inputs)
        self.h_means, self.h_vars = self.model.gating_network.predict_fs(
            self.test_inputs
        )
        self.mixing_probs = self.model.predict_mixing_probs(self.test_inputs)

    def plot_data(self, fig, ax):
        ax.scatter(
            self.X,
            self.Y,
            marker="x",
            color="k",
            alpha=0.4,
            lw=0.5,
            label="Observations",
        )

    def plot_gp(
        self, fig, ax, mean, var, Z=None, mu_label="", var_label="", color="C0"
    ):
        # label = "$\mathbb{E}[" + label + "_{" + str(k + 1) + "}(\mathbf{x}_*)]$"
        # label = "$\mathbb{V}[" + label + "_{" + str(k + 1) + "}(\mathbf{x}_*)]$"
        # ax.plot(self.test_inputs, mean, color=color, lw=2, label=mu_label)
        # ax.plot(self.test_inputs, mean, color=color, label=mu_label)
        ax.plot(self.test_inputs, mean, label=mu_label, color=color)
        ax.fill_between(
            self.test_inputs[:, 0],
            mean - 1.96 * np.sqrt(var),
            mean + 1.96 * np.sqrt(var),
            color=color,
            alpha=0.2,
            label=var_label,
        )
        if Z is not None:
            ax.scatter(Z, np.zeros(Z.shape), marker="|", color=color)

    def plot_dataset(self, save_filename=None):
        fig = plt.figure(figsize=(self.figsize[0] / 2, self.figsize[1] / 2))
        gs = fig.add_gridspec(1, 1)
        ax = gs.subplots()
        ax.quiver(self.X[:, 0], self.X[:, 1], self.Y[:, 0], self.Y[:, 1])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_model(self, save_dir):
        save_filename = os.path.join(save_dir, "y_samples.pdf")
        self.plot_y_samples(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "experts_y.pdf")
        self.plot_experts_y(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "experts_f.pdf")
        self.plot_experts_f(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "gating_gps.pdf")
        self.plot_gating_gps(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "gating_mixing_probs.pdf")
        self.plot_mixing_probs(save_filename=save_filename)
        # save_filename = os.path.join(save_dir, "dataset_quiver.pdf")
        # self.plot_dataset(save_filename=save_filename)

    def plot_y_samples(self, save_filename=None):
        fig, axs = self.create_fig_axs_plot_y_samples()
        self.plot_y_samples_given_fig_axs(fig, axs)
        if save_filename is not None:
            # plt.savefig(save_filename, transparent=True)
            plt.savefig(save_filename)

    def plot_experts_y(self, save_filename=None):
        fig, axs = self.create_fig_axs_plot_experts_f()
        self.plot_experts_y_given_fig_axs(fig, axs)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_experts_f(self, save_filename=None):
        fig, axs = self.create_fig_axs_plot_experts_f()
        self.plot_experts_f_given_fig_axs(fig, axs)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_gating_gps(self, save_filename=None):
        fig, axs = self.create_fig_axs_plot_gating_gps()
        self.plot_gating_gps_given_fig_axs(fig, axs)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_mixing_probs(self, save_filename=None):
        fig, axs = self.create_fig_axs_plot_mixing_probs()
        self.plot_mixing_probs_given_fig_axs(fig, axs)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def create_fig_axs_plot_y_samples(self):
        # fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] / 2))
        # gs = fig.add_gridspec(1, 1, wspace=0.3)
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_experts_f(self):
        fig = plt.figure(figsize=self.figsize)
        # gs = fig.add_gridspec(1, 1, wspace=0.3)
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_gating_gps(self):
        # fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] / 2))
        # gs = fig.add_gridspec(2, 2, wspace=0.3)
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_mixing_probs(self):
        # fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] / 4))
        fig = plt.figure(figsize=self.figsize)
        # gs = fig.add_gridspec(1, 2, wspace=0.3)
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots(sharex=True, sharey=True)
        # axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def plot_y_samples_given_fig_axs(self, fig, ax):
        # svgp_color = "k"
        # svgp_color = "magenta"
        # svgp_color = "r"
        color_idx = self.num_experts
        svgp_color = colors[color_idx]
        # y_color = colors[color_idx]
        # y_color = "darkred"
        # y_color = "cyan"
        # y_color = "k"
        # y_color = "y"
        y_linestyle = "--"
        lw = 0.7
        # ax.set_facecolor("lightgrey")
        self.test_inputs_broadcast = np.broadcast_to(
            self.test_inputs, shape=self.y_samples.shape
        )
        z = self.y_samples_dist.prob(self.y_samples)
        # ax.plot(
        #     self.test_inputs,
        #     self.svgp_mean,
        #     color=svgp_color,
        #     lw=lw,
        #     label="SVGP predictive mean $\mathbb{E}_{\\text{SVGP}}[y \mid x]$",
        # )
        ax.fill_between(
            self.test_inputs[:, 0],
            self.svgp_mean[:, 0] - 1.96 * np.sqrt(self.svgp_var[:, 0]),
            self.svgp_mean[:, 0] + 1.96 * np.sqrt(self.svgp_var[:, 0]),
            color=svgp_color,
            alpha=0.2,
            label="SVGP $\pm 2\sigma$"
            # label="SVGP $\pm 2\sigma$ with $\mathcal{N}(y_* \mid \mu_*, \sigma^2_*)$",
        )
        scatter = ax.scatter(
            self.test_inputs_broadcast,
            self.y_samples,
            c=z,
            # s=9,
            s=3,
            # cmap=self.cmap,
            # cmap="Greys",
            # rasterized=True,
            alpha=0.8,
            label="MoSVGPE samples",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("$p(y \mid x)$")

        # ax.plot(
        #     self.test_inputs,
        #     self.y_mean,
        #     color=y_color,
        #     lw=lw,
        #     linestyle=y_linestyle,
        #     label="Our predictive mean $\mathbb{E}_{\\text{ours}}[y \mid x]$",
        # )

        # ax.plot(
        #     self.test_inputs,
        #     self.svgp_mean,
        #     lw=lw,
        #     color=svgp_color,
        #     label="SVGP predictive mean $\mathbb{E}_{\\text{SVGP}}[y \mid x]$",
        # )
        ax.plot(
            self.test_inputs,
            self.svgp_mean - 1.96 * np.sqrt(self.svgp_var),
            color=svgp_color,
            lw=lw,
        )
        ax.plot(
            self.test_inputs,
            self.svgp_mean + 1.96 * np.sqrt(self.svgp_var),
            color=svgp_color,
            lw=lw,
        )
        ax.legend(loc=2)

    def plot_experts_y_given_fig_axs(self, fig, axs):
        self.plot_experts_given_fig_axs(fig, axs, self.y_means, self.y_vars, label="y")

    def plot_experts_f_given_fig_axs(self, fig, axs):
        self.plot_experts_given_fig_axs(fig, axs, self.f_means, self.f_vars, label="f")

    def plot_experts_given_fig_axs(self, fig, ax, means, vars, label="f"):
        row = 0
        for k in range(self.num_experts):
            # mu_label = "$\mathbb{E}[" + label + "_{" + str(k + 1) + "}(\mathbf{x}_*)]$"
            # var_label = "$\mathbb{V}[" + label + "_{" + str(k + 1) + "}(\mathbf{x}_*)]$"
            mu_label = "$k=" + str(k + 1) + "$"
            var_label = None
            Z = self.model.experts.experts_list[k].inducing_variable.Z
            self.plot_gp(
                fig,
                ax,
                means[:, 0, k],
                vars[:, 0, k],
                Z=Z,
                mu_label=mu_label,
                var_label=var_label,
                color=expert_colors[k],
            )
        self.plot_data(fig, ax)
        ax.legend()

    def plot_gating_gps_given_fig_axs(self, fig, ax):
        for k in range(self.num_experts):
            try:
                Z = self.model.gating_network.inducing_variable.Z
            except AttributeError:
                try:
                    Z = self.model.gating_network.inducing_variable.inducing_variables[
                        0
                    ].Z
                except AttributeError:
                    Z = self.model.gating_network.inducing_variable.inducing_variables[
                        0
                    ].Z
            # mu_label = "$\mathbb{E}[h_{" + str(k + 1) + "}(\mathbf{x}_*)]$"
            # var_label = "$\mathbb{V}[h_{" + str(k + 1) + "}(\mathbf{x}_*)]$"
            mu_label = "$k=" + str(k + 1) + "$"
            var_label = None
            self.plot_gp(
                fig,
                ax,
                self.h_means[:, k],
                self.h_vars[:, k],
                Z=Z,
                mu_label=mu_label,
                var_label=var_label,
                color=expert_colors[k],
            )
        ax.legend()

    def plot_mixing_probs_given_fig_axs(self, fig, ax):
        # ax.set_ylabel = "$\Pr(\\alpha_* = k \mid \mathbf{x}_*)$"
        ax.set(xlabel="$x$", ylabel="$\Pr(\\alpha_* = k \mid \mathbf{x}_*)$")
        for k in range(self.num_experts):
            ax.plot(
                self.test_inputs,
                self.mixing_probs[:, k],
                label="$k=" + str(k + 1) + "$",
                color=expert_colors[k],
            )
        ax.legend()


if __name__ == "__main__":
    two_expert_ckpt_dir = (
        "./mcycle/saved_ckpts/2_experts/batch_size_32/learning_rate_0.01/10-06-100402"
    )
    three_expert_ckpt_dir = (
        "./mcycle/saved_ckpts/3_experts/batch_size_32/learning_rate_0.01/10-06-105414"
    )

    # Load config (with model and training params) from toml file
    two_expert_config_file = (
        "./mcycle/configs/config_2_experts_subset.toml"  # path to config
    )
    three_expert_config_file = (
        "./mcycle/configs/config_3_experts_subset.toml"  # path to config
    )
    two_expert_cfg = config_from_toml(two_expert_config_file, read_from_file=True)
    three_expert_cfg = config_from_toml(three_expert_config_file, read_from_file=True)

    # Load mcycle data set
    try:
        standardise = two_expert_cfg.standardise
    except AttributeError:
        standardise = False
    dataset = load_mcycle_dataset(
        two_expert_cfg.data_file, plot=False, standardise=standardise
    )

    two_expert_model = load_model_from_config_and_checkpoint(
        two_expert_config_file, two_expert_ckpt_dir, dataset
    )
    three_expert_model = load_model_from_config_and_checkpoint(
        three_expert_config_file, three_expert_ckpt_dir, dataset
    )

    two_expet_plotter = McyclePlotter(two_expert_model, X=dataset[0], Y=dataset[1])
    two_expet_plotter.plot_model("./mcycle/images/two-experts")

    three_expet_plotter = McyclePlotter(three_expert_model, X=dataset[0], Y=dataset[1])
    three_expet_plotter.plot_model("./mcycle/images/three-experts")
