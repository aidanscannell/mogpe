#!/usr/bin/env python3
import os

import numpy as np
import palettable
import tensorflow as tf
from gpflow.monitor import MonitorTaskGroup
from matplotlib import pyplot as plt
from mogpe.training.monitor import ImageWithCbarToTensorBoard
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plt.style.use("science")
# plt.style.use("ggplot")
plt.style.use("seaborn-paper")
# plt.style.use("seaborn")
# plt.style.use("seaborn-dark-palette")


def init_axis_labels_and_ticks(axs):
    for ax in axs.flat:
        ax.set(xlabel="$x$", ylabel="$y$")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    return axs


class QuadcopterPlotter:
    def __init__(
        self,
        model,
        X,
        Y,
        test_inputs=None,
        # num_samples=100,
        # params=None,
        # num_levels=6,
        figsize=(6.4, 4.8),
        cmap=palettable.scientific.sequential.Bilbao_15.mpl_colormap,
        static: bool = True,  # whether or not to recalculate model predictions at each call
    ):
        self.model = model
        self.X = X
        self.Y = Y
        self.num_experts = self.model.num_experts
        self.output_dim = Y.shape[1]
        self.figsize = figsize
        self.cmap = cmap
        self.static = static
        if test_inputs is None:
            num_test = 400
            factor = 1.2
            sqrtN = int(np.sqrt(num_test))
            xx = np.linspace(
                tf.reduce_min(X[:, 0]) * factor, tf.reduce_max(X[:, 0]) * factor, sqrtN
            )
            yy = np.linspace(
                tf.reduce_min(X[:, 1]) * factor, tf.reduce_max(X[:, 1]) * factor, sqrtN
            )
            xx, yy = np.meshgrid(xx, yy)
            self.test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        else:
            self.test_inputs = test_inputs

        self.y_mm_dist = self.model.predict_y(self.test_inputs)
        self.y_mm_means = self.y_mm_dist.mean()
        self.y_mm_vars = self.y_mm_dist.variance()
        self.y_means, self.y_vars = self.model.experts.predict_ys(self.test_inputs)
        self.f_means, self.f_vars = self.model.predict_experts_fs(self.test_inputs)
        self.h_means, self.h_vars = self.model.gating_network.predict_fs(
            self.test_inputs
        )
        self.mixing_probs = self.model.predict_mixing_probs(self.test_inputs)

    def contf(self, fig, ax, z):
        contf = ax.tricontourf(
            self.test_inputs[:, 0],
            self.test_inputs[:, 1],
            z,
            levels=10,
            # levels=var_levels,
            cmap=self.cmap,
        )
        return contf

    def plot_gp_contf(self, fig, axs, mean, var, mean_levels=None, var_levels=None):
        """Plots contours for mean and var side by side"""
        mean_contf = self.contf(fig, axs[0], z=mean)
        var_contf = self.contf(fig, axs[1], z=var)
        return mean_contf, var_contf

    def add_cbar(self, fig, ax, contf, label=""):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(contf, cax=cax)
        cbar.set_label(label)

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
        save_filename = os.path.join(save_dir, "y_moment_matched.pdf")
        self.plot_y_mm(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "experts_y.pdf")
        self.plot_experts_y(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "experts_f.pdf")
        self.plot_experts_f(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "gating_gps.pdf")
        self.plot_gating_gps(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "gating_mixing_probs.pdf")
        self.plot_mixing_probs(save_filename=save_filename)
        save_filename = os.path.join(save_dir, "dataset_quiver.pdf")
        self.plot_dataset(save_filename=save_filename)

    def plot_y_mm(self, save_filename=None):
        fig, axs = self.create_fig_axs_plot_y_mm()
        self.plot_y_mm_given_fig_axs(fig, axs)
        if save_filename is not None:
            plt.savefig(save_filename, transparent=True)

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

    def create_fig_axs_plot_y_mm(self):
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] / 2))
        gs = fig.add_gridspec(2, 2, wspace=0.3)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_experts_f(self):
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(4, 2, wspace=0.3)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_gating_gps(self):
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] / 2))
        gs = fig.add_gridspec(2, 2, wspace=0.3)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def create_fig_axs_plot_mixing_probs(self):
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] / 4))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        axs = gs.subplots(sharex=True, sharey=True)
        axs = init_axis_labels_and_ticks(axs)
        return fig, axs

    def plot_y_mm_given_fig_axs(self, fig, axs):
        if self.static:
            y_mm_means = self.y_mm_means
            y_mm_vars = self.y_mm_vars
        else:
            self.y_mm_dist = self.model.predict_y(self.test_inputs)
            y_mm_means = self.y_mm_dist.mean()
            y_mm_vars = self.y_mm_dist.variance()

        for dim in range(self.output_dim):
            mean_contf, var_contf = self.plot_gp_contf(
                fig,
                axs[dim, :],
                y_mm_means[:, dim],
                y_mm_vars[:, dim],
            )
            self.add_cbar(
                fig,
                axs[dim, 0],
                mean_contf,
                "$\mathbb{E}[\Delta\mathbf{x}_{" + str(dim + 1) + "}]$",
            )
            self.add_cbar(
                fig,
                axs[dim, 1],
                var_contf,
                "$\mathbb{V}[\Delta\mathbf{x}_{" + str(dim + 1) + "}]$",
            )

    def plot_experts_y_given_fig_axs(self, fig, axs):
        if self.static:
            y_means = self.y_means
            y_vars = self.y_vars
        else:
            y_means, y_vars = self.model.experts.predict_ys(self.test_inputs)
        self.plot_experts_given_fig_axs(fig, axs, y_means, y_vars, label="y")

    def plot_experts_f_given_fig_axs(self, fig, axs):
        if self.static:
            f_means = self.f_means
            f_vars = self.f_vars
        else:
            f_means, f_vars = self.model.predict_experts_fs(self.test_inputs)
        self.plot_experts_given_fig_axs(fig, axs, f_means, f_vars, label="f")

    def plot_experts_given_fig_axs(self, fig, axs, means, vars, label="f"):
        row = 0
        for k in range(self.num_experts):
            for j in range(self.output_dim):
                mean_contf, var_contf = self.plot_gp_contf(
                    fig,
                    axs[row, :],
                    means[:, j, k],
                    vars[:, j, k],
                )
                self.add_cbar(
                    fig,
                    axs[row, 0],
                    mean_contf,
                    "$\mathbb{E}["
                    + label
                    + "_{"
                    + str(k + 1)
                    + str(j + 1)
                    + "}(\mathbf{x}_*)]$",
                )
                self.add_cbar(
                    fig,
                    axs[row, 1],
                    var_contf,
                    "$\mathbb{V}["
                    + label
                    + "_{"
                    + str(k + 1)
                    + str(j + 1)
                    + "}(\mathbf{x}_*)]$",
                )
                row += 1

    def plot_gating_gps_given_fig_axs(self, fig, axs):
        if self.static:
            h_means = self.h_means
            h_vars = self.h_vars
        else:
            h_means, h_vars = self.model.gating_network.predict_fs(self.test_inputs)
        for k in range(self.num_experts):
            mean_contf, var_contf = self.plot_gp_contf(
                fig,
                axs[k, :],
                h_means[:, k],
                h_vars[:, k],
            )
            self.add_cbar(
                fig,
                axs[k, 0],
                mean_contf,
                "$\mathbb{E}[h_{" + str(k + 1) + "}(\mathbf{x}_*)]$",
            )
            self.add_cbar(
                fig,
                axs[k, 1],
                var_contf,
                "$\mathbb{V}[h_{" + str(k + 1) + "}(\mathbf{x}_*)]$",
            )

    def plot_mixing_probs_given_fig_axs(self, fig, axs):
        if self.static:
            mixing_probs = self.mixing_probs
        else:
            mixing_probs = self.model.predict_mixing_probs(self.test_inputs)
        for k in range(self.num_experts):
            prob_contf = self.contf(fig, axs[k], z=mixing_probs[:, k])
            self.add_cbar(
                fig,
                axs[k],
                prob_contf,
                "$\Pr(\\alpha_* = " + str(k + 1) + " \mid \mathbf{x}_*)$",
            )

    def plot_gating_network_given_fig_axs(self, fig, axs):
        if self.static:
            mixing_probs = self.mixing_probs
        else:
            mixing_probs = self.model.predict_mixing_probs(self.test_inputs)
        for k in range(self.num_experts):
            prob_contf = self.contf(fig, axs[k], z=mixing_probs[:, k])
            self.add_cbar(
                fig,
                axs[k],
                prob_contf,
                "$\Pr(\\alpha_* = " + str(k + 1) + " \mid \mathbf{x}_*)$",
            )

    def tf_monitor_task_group(self, log_dir, slow_tasks_period=500):
        image_task_experts_f = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_experts_f_given_fig_axs,
            name="experts_latent_function_posterior",
            fig_kw={"figsize": self.figsize},
            subplots_kw={
                "nrows": self.num_experts * self.output_dim,
                "ncols": 2,
                # "wspace": 0.3,
                "sharex": True,
                "sharey": True,
            },
        )
        image_task_experts_y = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_experts_y_given_fig_axs,
            name="experts_output_posterior",
            fig_kw={"figsize": self.figsize},
            subplots_kw={
                "nrows": self.num_experts * self.output_dim,
                "ncols": 2,
                # "wspace": 0.3,
                "sharex": True,
                "sharey": True,
            },
        )
        image_task_gating_gps = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_gating_gps_given_fig_axs,
            name="gating_network_gps_posteriors",
            fig_kw={"figsize": (self.figsize[0], self.figsize[1] / 2)},
            subplots_kw={
                "nrows": self.num_experts,
                "ncols": 2,
                # "wspace": 0.3,
                "sharex": True,
                "sharey": True,
            },
        )
        image_task_mixing_probs = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_mixing_probs_given_fig_axs,
            name="gating_network_mixing_probabilities",
            fig_kw={"figsize": (self.figsize[0], self.figsize[1] / 4)},
            subplots_kw={
                "nrows": 1,
                "ncols": self.num_experts,
                "sharex": True,
                "sharey": True,
            },
        )
        image_task_y = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_y_mm_given_fig_axs,
            name="predictive_posterior_moment_matched",
            fig_kw={"figsize": (self.figsize[0], self.figsize[1] / 2)},
            subplots_kw={
                "nrows": self.output_dim,
                "ncols": 2,
                # "wspace": 0.3,
                "sharex": True,
                "sharey": True,
            },
        )
        image_tasks = [
            image_task_experts_y,
            image_task_experts_f,
            image_task_mixing_probs,
            image_task_gating_gps,
            image_task_y,
        ]
        return MonitorTaskGroup(image_tasks, period=slow_tasks_period)
