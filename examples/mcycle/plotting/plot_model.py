#!/usr/bin/env python3
import os

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import palettable
import tensorflow as tf
import tensorflow_probability as tfp
from config import config_from_toml
from gpflow.likelihoods import Bernoulli, Softmax
from matplotlib.colors import LinearSegmentedColormap
from mcycle.data.load_data import load_mcycle_dataset
from mcycle.train_svgp_on_mcycle_from_config import init_svgp_from_config
from mogpe.training import load_model_from_config_and_checkpoint
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mogpe.training.metrics import (
    negative_log_predictive_density,
    root_mean_squared_error,
    mean_absolute_error,
)
from mcycle.plotting.plot_metrics import restore_svgp

tfd = tfp.distributions
# plt.style.use("science")
plt.style.use("seaborn-paper")
plt.style.use("ggplot")

# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

# plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

# expert_colors = ["r", "g", "b"]
expert_colors = ["c", "m", "y"]
# svgp_color = "o"

# svgp_color = colors[4]  # num_experts+1
# svgp_color = "y"
svgp_color = "m"
svgp_color = "r"
# svgp_color = "purple"
# svgp_color = "crimson"
# svgp_color = "crimson"
# svgp_color = "olive"
# svgp_color = "lime"
svgp_linestyle = "--"
# svgp_linestyle = "-"

y_color = "blue"
y_color = "black"
# y_color = "lime"
# y_color = "c"
y_linestyle = "-"


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
        svgp=None,
        test_inputs=None,
        # num_samples=100,
        # num_samples=50,
        num_samples=80,
        # params=None,
        # num_levels=6,
        figsize=(6.4, 4.8),
        # figsize=(6.4 / 2, 4.8 / 2),
        cmap=palettable.scientific.sequential.Bilbao_15.mpl_colormap,
    ):
        self.model = model
        self.X = X
        self.Y = Y
        self.svgp = svgp
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
        self.y_var = self.model.predict_y(self.test_inputs).variance()
        # self.y_samples = self.y_samples_dist.sample(self.num_samples)
        # self.test_inputs_broadcast = np.expand_dims(self.test_inputs, 0)
        if svgp is None:
            self.svgp_mean, self.svgp_var = self.model.experts.predict_ys(
                self.test_inputs
            )
            self.svgp_mean = self.svgp_mean[:, :, 1]
            self.svgp_var = self.svgp_var[:, :, 1]
        else:
            self.svgp_mean, self.svgp_var = self.svgp.predict_y(self.test_inputs)

        self.y_means, self.y_vars = self.model.experts.predict_ys(self.test_inputs)
        self.f_means, self.f_vars = self.model.predict_experts_fs(self.test_inputs)
        self.h_means, self.h_vars = self.model.gating_network.predict_fs(
            self.test_inputs
        )
        self.mixing_probs = self.model.predict_mixing_probs(self.test_inputs)

        # Generate samples from full model
        if isinstance(self.model.gating_network.likelihood, Softmax):
            self.alpha_dist = tfd.Categorical(probs=self.mixing_probs)
        elif isinstance(self.model.gating_network.likelihood, Bernoulli):
            self.alpha_dist = tfd.Bernoulli(probs=self.mixing_probs[:, 1])

        self.alpha_samples = self.alpha_dist.sample(num_samples)
        self.alpha_samples = tf.expand_dims(self.alpha_samples, -1)
        self.experts_dists = tfd.Normal(self.y_means, self.y_vars)
        self.experts_samples = self.experts_dists.sample(self.num_samples)
        self.y_samples = self.experts_samples[:, :, :, 0]
        for k in range(self.num_experts):
            self.y_samples = tf.where(
                self.alpha_samples == k,
                self.experts_samples[:, :, :, k],
                self.y_samples,
            )
        self.test_inputs_broadcast = np.broadcast_to(
            self.test_inputs, shape=self.y_samples.shape
        )

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
        save_filename = os.path.join(save_dir, "y_means.pdf")
        self.plot_y_means(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "y_samples_density.pdf")
        self.plot_y_samples_density(save_filename=save_filename)
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

    def plot_y_means(self, save_filename=None):
        fig, axs = self.create_fig_axs_plot_y_means()
        self.plot_y_means_given_fig_axs(fig, axs)
        if save_filename is not None:
            # plt.savefig(save_filename, transparent=True)
            plt.savefig(save_filename)

    def plot_y_samples_density(self, save_filename=None):
        fig, ax = self.create_fig_axs_plot_y_samples_density()
        ax.set_ylim(-3.5, 3.5)
        self.plot_y_samples_density_given_fig_axs(fig, ax)
        if save_filename is not None:
            # plt.savefig(save_filename, transparent=True)
            plt.savefig(save_filename)

    def plot_y_samples(self, save_filename=None):
        fig, ax = self.create_fig_axs_plot_y_samples()
        ax.set_ylim(-3.5, 3.5)
        self.plot_y_samples_given_fig_axs(fig, ax)
        if save_filename is not None:
            # plt.savefig(save_filename, transparent=True)
            plt.savefig(save_filename)

    def plot_experts_y(self, save_filename=None):
        fig, ax = self.create_fig_axs_plot_experts_f()
        ax.set_ylim(-3.5, 3.5)
        self.plot_experts_y_given_fig_axs(fig, ax)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

    def plot_experts_f(self, save_filename=None):
        fig, ax = self.create_fig_axs_plot_experts_f()
        ax.set_ylim(-3.5, 3.5)
        self.plot_experts_f_given_fig_axs(fig, ax)
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

    def create_fig_axs_plot_y_means(self):
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_y_samples_density(self):
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_y_samples(self):
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_experts_f(self):
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_gating_gps(self):
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

    def plot_y_means_given_fig_axs(self, fig, ax):
        lw = 0.7
        self.plot_data(fig, ax)
        ax.plot(
            self.test_inputs,
            self.svgp_mean,
            lw=lw,
            color=svgp_color,
            linestyle=svgp_linestyle,
            label="SVGP mean",
            # label="SVGP predictive mean $\mathbb{E}_{\\text{SVGP}}[y \mid x]$",
        )

        ax.plot(
            self.test_inputs,
            self.y_mean,
            color=y_color,
            lw=lw,
            linestyle=y_linestyle,
            label="MoSVGPE mean"
            # label="Our predictive mean $\mathbb{E}_{\\text{ours}}[y \mid x]$",
        )
        ax.legend(loc=2)

    def plot_y_samples_density_given_fig_axs(self, fig, ax):
        lw = 0.7
        lw = 1.1
        # ax.fill_between(
        #     self.test_inputs[:, 0],
        #     self.svgp_mean[:, 0] - 1.96 * np.sqrt(self.svgp_var[:, 0]),
        #     self.svgp_mean[:, 0] + 1.96 * np.sqrt(self.svgp_var[:, 0]),
        #     color=svgp_color,
        #     alpha=0.2,
        #     label="SVGP $\pm 2\sigma$",
        # )
        scatter = ax.scatter(
            self.test_inputs_broadcast,
            self.y_samples,
            c=self.y_samples_dist.prob(self.y_samples),
            # s=9,
            s=3,
            # cmap=self.cmap,
            # rasterized=True,
            alpha=0.8,
            label="MoSVGPE samples",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("$p(y \mid x)$")
        ax.plot(
            self.test_inputs,
            self.svgp_mean - 1.96 * np.sqrt(self.svgp_var),
            color=svgp_color,
            lw=lw,
            label="SVGP $\pm 2\sigma$",
        )
        ax.plot(
            self.test_inputs,
            self.svgp_mean + 1.96 * np.sqrt(self.svgp_var),
            color=svgp_color,
            lw=lw,
        )
        ax.legend(loc=2)

    def plot_y_samples_given_fig_axs(self, fig, ax):
        lw = 0.7
        lw = 1.1
        # ax.fill_between(
        #     self.test_inputs[:, 0],
        #     self.svgp_mean[:, 0] - 1.96 * np.sqrt(self.svgp_var[:, 0]),
        #     self.svgp_mean[:, 0] + 1.96 * np.sqrt(self.svgp_var[:, 0]),
        #     color=svgp_color,
        #     alpha=0.2,
        #     label="SVGP $\pm 2\sigma$",
        # )
        cmap = LinearSegmentedColormap.from_list(
            "expert_colors",
            colors=expert_colors[: self.num_experts],
            N=self.num_experts,
        )
        for k in range(self.num_experts):
            ax.scatter(
                self.test_inputs_broadcast[self.alpha_samples == k],
                self.y_samples[self.alpha_samples == k],
                # c=self.alpha_samples[self.alpha_samples == k],
                c=expert_colors[k],
                # s=9,
                s=3,
                cmap=cmap,
                # rasterized=True,
                alpha=0.8,
                label="k=" + str(k + 1) + " samples",
            )
        # ax.scatter(
        #     self.test_inputs_broadcast,
        #     self.y_samples,
        #     c=self.alpha_samples,
        #     # s=9,
        #     s=3,
        #     cmap=cmap,
        #     # rasterized=True,
        #     alpha=0.8,
        #     label="MoSVGPE samples",
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
            label="SVGP $\pm 2\sigma$",
        )
        # ax.legend(loc=2)
        ax.legend(facecolor="gray")
        ax.legend(loc=3)

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
                # Z=Z,
                Z=None,
                mu_label=mu_label,
                var_label=var_label,
                color=expert_colors[k],
            )
        ax.set_ylabel("$" + label + "_k(x)$")
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
                        k
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
                # Z=Z,
                Z=None,
                mu_label=mu_label,
                var_label=var_label,
                color=expert_colors[k],
            )
        ax.set_ylabel("$h_k(x)$")
        ax.legend()

    def plot_mixing_probs_given_fig_axs(self, fig, ax):
        # ax.set_ylabel = "$\Pr(\\alpha_* = k \mid \mathbf{x}_*)$"
        ax.set(xlabel="$x$", ylabel="$\Pr(\\alpha = k \mid x)$")
        for k in range(self.num_experts):
            ax.plot(
                self.test_inputs,
                self.mixing_probs[:, k],
                label="$k=" + str(k + 1) + "$",
                color=expert_colors[k],
            )
        ax.legend()


if __name__ == "__main__":

    # ckpt_dirs = {
    #     "K=2_L1": "./logs/mcycle/full-dataset/2_experts/batch_size_16/learning_rate_0.01/tight_bound/num_inducing_32/10-20-141622",
    #     # "K=2_L1": "./mcycle/saved_ckpts/full-dataset/2_experts/batch_size_16/learning_rate_0.01/tight_bound/num_inducing_32/10-15-140247",
    #     "K=2_L2": "./mcycle/saved_ckpts/full-dataset/2_experts/batch_size_16/learning_rate_0.01/further_bound/num_inducing_32/10-15-140200",
    #     "K=3_L1": "./mcycle/saved_ckpts/full-dataset/3_experts/batch_size_16/learning_rate_0.01/tight_bound/num_inducing_32/10-15-140858",
    #     "K=3_L2": "./mcycle/saved_ckpts/full-dataset/3_experts/batch_size_16/learning_rate_0.01/further_bound/num_inducing_32/10-15-140829",
    # }
    ckpt_dirs = {
        "K=2_L1": "./logs/mcycle/full-dataset/2_experts/batch_size_16/learning_rate_0.01/tight_bound/num_inducing_32/10-20-141622",
        "K=2_L2": "./mcycle/saved_ckpts/full-dataset/2_experts/batch_size_16/learning_rate_0.01/further_gating_bound/num_inducing_32/10-26-112847",
        "K=2_L3": "./mcycle/saved_ckpts/full-dataset/2_experts/batch_size_16/learning_rate_0.01/further_bound/num_inducing_32/10-15-140200",
        "K=3_L1": "./mcycle/saved_ckpts/full-dataset/3_experts/batch_size_16/learning_rate_0.01/tight_bound/num_inducing_32/10-15-140858",
        "K=3_L2": "./mcycle/saved_ckpts/full-dataset/3_experts/batch_size_16/learning_rate_0.01/further_gating_bound/num_inducing_32/10-26-112927",
        "K=3_L3": "./mcycle/saved_ckpts/full-dataset/3_experts/batch_size_16/learning_rate_0.01/further_bound/num_inducing_32/10-15-140829",
    }
    configs = {
        "K=2_L1": "./mcycle/configs/config_2_experts_full.toml",
        "K=2_L2": "./mcycle/configs/config_2_experts_full.toml",
        "K=2_L3": "./mcycle/configs/config_2_experts_full.toml",
        "K=3_L1": "./mcycle/configs/config_3_experts_full.toml",
        "K=3_L2": "./mcycle/configs/config_3_experts_full.toml",
        "K=3_L3": "./mcycle/configs/config_3_experts_full.toml",
    }

    # SVGP model to use
    svgp_config = "./mcycle/configs/config_svgp_m_32_full.toml"
    svgp_ckpt_dir = "./mcycle/saved_ckpts/full-dataset/svgp/1_experts/batch_size_16/learning_rate_0.01/normal_bound/num_inducing_32/10-15-145648"
    svgp_model = restore_svgp(svgp_config, svgp_ckpt_dir)

    # Load mcycle data set
    cfg = config_from_toml(svgp_config, read_from_file=True)
    train_dataset, test_dataset = load_mcycle_dataset(
        cfg.data_file, plot=False, standardise=cfg.standardise
    )

    svgp_rmse = root_mean_squared_error(
        svgp_model, dataset=train_dataset, batched=True
    ).numpy()

    results = {}
    for model_str in ckpt_dirs:
        cfg = config_from_toml(configs[model_str], read_from_file=True)

        # Restore MoSVGPE checkpoint
        model = load_model_from_config_and_checkpoint(
            configs[model_str], ckpt_dirs[model_str], dataset=train_dataset
        )
        rmse = root_mean_squared_error(
            model, dataset=train_dataset, batched=True
        ).numpy()
        # rmse = root_mean_squared_error(model, dataset=(X, Y)).numpy()

        # Plot model
        plotter = McyclePlotter(
            model, X=train_dataset[0], Y=train_dataset[1], svgp=svgp_model
        )
        # fig, axs = plotter.create_fig_axs_plot_y_means()
        # plotter.plot_y_means_given_fig_axs(fig, axs)
        # axs.plot(train_dataset[0], svgp_rmse - rmse, color="b") * 100
        # diff = svgp_rmse - rmse
        # print(diff)
        # axs.plot(train_dataset[0], rmse, color="b")
        # axs.plot(train_dataset[0], svgp_rmse, color="r")
        # plt.show()

        plotter.plot_model("./mcycle/images/" + model_str)
