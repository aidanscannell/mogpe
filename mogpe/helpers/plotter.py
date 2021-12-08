from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import palettable
import tensorflow as tf
from gpflow.monitor import ImageToTensorBoard, MonitorTaskGroup
from mogpe.training.monitor import ImageWithCbarToTensorBoard
from mpl_toolkits.axes_grid1 import make_axes_locatable

color_1 = "olive"
color_2 = "darkmagenta"
color_2 = "darkslategrey"
color_3 = "darkred"
color_3 = "lime"
color_obs = "red"

plt.style.use("science")
plt.style.use("seaborn-paper")

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


class PlotterBase(ABC):
    def __init__(self, model, X, Y, num_samples=100, params=None):
        self.num_samples = num_samples
        self.model = model
        self.num_experts = self.model.num_experts
        self.X = X
        self.Y = Y
        self.output_dim = Y.shape[1]
        if params is None:
            params = {
                # 'axes.labelsize': 30,
                # 'font.size': 30,
                # 'legend.fontsize': 20,
                # 'xtick.labelsize': 30,
                # 'ytick.labelsize': 30,
                # 'text.usetex': True,
            }
        plt.rcParams.update(params)

    @abstractmethod
    def plot_gp(self, fig, ax, mean, var):
        raise NotImplementedError

    @abstractmethod
    def plot_experts_f(self, fig, ax):
        raise NotImplementedError

    @abstractmethod
    def plot_experts_y(self, fig, ax):
        raise NotImplementedError

    @abstractmethod
    def plot_gating_network(self, fig, ax):
        raise NotImplementedError

    @abstractmethod
    def plot_y(self, fig, ax):
        raise NotImplementedError

    @abstractmethod
    def tf_monitor_task_group(self, log_dir, slow_period=500):
        raise NotImplementedError


class Plotter1D(PlotterBase):
    def __init__(self, model, X, Y, test_inputs=None, num_samples=100, params=None):
        super().__init__(model, X, Y, num_samples, params)
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

    def plot_gp(self, fig, ax, mean, var, label=""):
        alpha = 0.4
        ax.scatter(self.X, self.Y, marker="x", color="k", alpha=alpha)
        ax.plot(self.test_inputs, mean, "C0", lw=2, label=label)
        ax.fill_between(
            self.test_inputs[:, 0],
            mean - 1.96 * np.sqrt(var),
            mean + 1.96 * np.sqrt(var),
            color="C0",
            alpha=0.2,
        )

    def plot_experts_f(self, fig, axs):
        # tf.print("Plotting experts f...")
        row = 0
        for k, expert in enumerate(self.model.experts.experts_list):
            mean, var = expert.predict_f(self.test_inputs)
            f_samples = expert.predict_f_samples(self.test_inputs, 5)
            Z = expert.inducing_variable.Z
            for j in range(self.output_dim):
                for f_sample in f_samples:
                    axs[row].plot(self.test_inputs, f_sample, alpha=0.3)
                self.plot_gp(fig, axs[row], mean[:, j], var[:, j])
                axs[row].scatter(Z, np.zeros(Z.shape), marker="|", color="k")
                row += 1

    def plot_experts_y(self, fig, axs):
        # tf.print("Plotting experts y...")
        dists = self.model.predict_experts_dists(self.test_inputs)
        mean = dists.mean()
        var = dists.variance()
        row = 0
        for k, expert in enumerate(self.model.experts.experts_list):
            for j in range(self.output_dim):
                self.plot_gp(fig, axs[row], mean[:, j, k], var[:, j, k])
                row += 1

    def plot_gating_network(self, fig, ax):
        # tf.print("Plotting gating network mixing probabilities...")
        mixing_probs = self.model.predict_mixing_probs(self.test_inputs)
        num_experts = tf.shape(mixing_probs)[-1]
        for k in range(num_experts):
            ax.plot(self.test_inputs, mixing_probs[:, k], label=str(k + 1))

        # Z = self.model.gating_network.inducing_variable.Z
        # ax.scatter(Z, np.zeros(Z.shape), marker="|")
        ax.legend()

    def plot_gating_gps(self, fig, axs):
        """Plots mean and var of gating network gp

        :param axs: if num_experts > 2: [num_experts, 2] else [1, 2]
        """
        # tf.print("Plotting gating network gps...")
        means, vars = self.model.gating_network.predict_fs(self.test_inputs)
        # Z = self.model.gating_network.inducing_variable.Z
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
            self.plot_gp(fig, axs[k], means[:, k], vars[:, k], label=str(k + 1))
            axs[k].scatter(Z, np.zeros(Z.shape), marker="|")
            axs[k].legend()

    def plot_samples(self, fig, ax, input_broadcast, y_samples, color=color_3):
        ax.scatter(
            input_broadcast,
            y_samples,
            marker=".",
            s=4.9,
            color=color,
            lw=0.4,
            rasterized=True,
            alpha=0.2,
        )

    def plot_y(self, fig, ax):
        # tf.print("Plotting y...")
        alpha = 0.4
        ax.scatter(self.X, self.Y, marker="x", color="k", alpha=alpha)
        y_dist = self.model.predict_y(self.test_inputs)
        y_samples = y_dist.sample(self.num_samples)
        ax.plot(self.test_inputs, y_dist.mean(), color="k")

        self.test_inputs_broadcast = np.expand_dims(self.test_inputs, 0)

        for i in range(self.num_samples):
            self.plot_samples(fig, ax, self.test_inputs_broadcast, y_samples[i, :, :])

    def plot_model(self):
        fig, ax = plt.subplots(1, 1)
        self.plot_gating_network(fig, ax)
        # fig, ax = plt.subplots(1, 1)
        # fig, axs = plt.subplots(1, self.num_experts, figsize=(10, 4))
        # self.plot_gating_gps(fig, ax)
        fig, axs = plt.subplots(1, self.num_experts, figsize=(10, 4))
        self.plot_experts_f(fig, axs)
        fig, ax = plt.subplots(1, 1)
        self.plot_y(fig, ax)

    def tf_monitor_task_group(self, log_dir, slow_period=500):
        ncols = 1
        nrows_experts = self.num_experts
        nrows_y = self.output_dim
        image_task_experts_f = ImageToTensorBoard(
            log_dir,
            self.plot_experts_f,
            name="experts_latent_function_posterior",
            fig_kw={"figsize": (10, 4)},
            subplots_kw={"nrows": nrows_experts, "ncols": self.output_dim},
        )
        image_task_experts_y = ImageToTensorBoard(
            log_dir,
            self.plot_experts_y,
            name="experts_output_posterior",
            fig_kw={"figsize": (10, 4)},
            subplots_kw={"nrows": nrows_experts, "ncols": self.output_dim},
        )
        image_task_gating_gps = ImageToTensorBoard(
            log_dir,
            self.plot_gating_gps,
            name="gating_network_gps_posteriors",
            subplots_kw={"nrows": nrows_experts, "ncols": 1},
        )
        image_task_gating = ImageToTensorBoard(
            log_dir,
            self.plot_gating_network,
            name="gating_network_mixing_probabilities",
            subplots_kw={"nrows": 1, "ncols": 1},
        )
        image_task_y = ImageToTensorBoard(
            log_dir, self.plot_y, name="predictive_posterior_samples"
        )
        # image_tasks = [
        #     image_task_experts_y, image_task_experts_f, image_task_gating
        # ]
        image_tasks = [
            image_task_experts_y,
            image_task_experts_f,
            image_task_gating_gps,
            image_task_gating,
            image_task_y,
        ]
        return MonitorTaskGroup(image_tasks, period=slow_period)


class Plotter2D(PlotterBase):
    def __init__(
        self,
        model,
        X,
        Y,
        test_inputs=None,
        num_samples=100,
        params=None,
        num_levels=6,
        cmap=palettable.scientific.sequential.Bilbao_15.mpl_colormap,
    ):
        super().__init__(model, X, Y, num_samples, params)
        self.cmap = cmap
        self.num_levels = num_levels
        # self.levels = np.linspace(0.0, 1.0, num_levels)
        self.levels = np.linspace(0.0, 1.0, 50)
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

    def plot_gp(self, fig, axs, mean, var, mean_levels=None, var_levels=None):
        """Plots contours and colorbars for mean and var side by side

        :param axs: [2,]
        :param mean: [num_data, 1]
        :param var: [num_data, 1]
        :param mean_levels: levels for mean contourf e.g. np.linspace(0, 1, 10)
        :param var_levels: levels for var contourf e.g. np.linspace(0, 1, 10)
        """
        mean_contf, var_contf = self.plot_gp_contf(
            fig, axs, mean, var, mean_levels, var_levels
        )
        mean_cbar = self.cbar(fig, axs[0], mean_contf)
        var_cbar = self.cbar(fig, axs[1], var_contf)
        return np.array([mean_cbar, var_cbar])

    def plot_gp_contf(self, fig, axs, mean, var, mean_levels=None, var_levels=None):
        """Plots contours for mean and var side by side

        :param axs: [2,]
        :param mean: [num_data, 1]
        :param var: [num_data, 1]
        :param mean_levels: levels for mean contourf e.g. np.linspace(0, 1, 10)
        :param var_levels: levels for var contourf e.g. np.linspace(0, 1, 10)
        """
        for ax in axs:
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
        mean_contf = axs[0].tricontourf(
            self.test_inputs[:, 0],
            self.test_inputs[:, 1],
            mean,
            100,
            # levels=mean_levels,
            cmap=self.cmap,
        )
        var_contf = axs[1].tricontourf(
            self.test_inputs[:, 0],
            self.test_inputs[:, 1],
            var,
            100,
            # levels=var_levels,
            cmap=self.cmap,
        )
        return mean_contf, var_contf

    def cbar(self, fig, ax, contf):
        """Adds cbar to ax or ax[0] is np.ndarray

        :param ax: either a matplotlib ax or np.ndarray axs
        :param contf: contourf
        """
        if isinstance(ax, np.ndarray):
            divider = make_axes_locatable(ax[0])
        else:
            divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = fig.colorbar(
            contf,
            ax=ax,
            use_gridspec=True,
            cax=cax,
            # format="%0.2f",
            orientation="horizontal",
        )
        # cbar.ax.locator_params(nbins=9)

        # cax.ticklabel_format(style="sci", scilimits=(0, 3))
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        return cbar

    def create_levels(self, means, vars):
        if tf.math.reduce_max(means) > 0.0:
            factor_mean = 1.01
        else:
            factor_mean = -1.01
        if tf.math.reduce_max(vars) > 0.0:
            factor_var = 1.01
        else:
            factor_var = -1.01
        mean_levels = tf.linspace(
            tf.math.reduce_min(means),
            tf.math.reduce_max(means) * factor_mean,
            self.num_levels,
        )
        var_levels = tf.linspace(
            tf.math.reduce_min(vars),
            tf.math.reduce_max(vars) * factor_var,
            self.num_levels,
        )
        return mean_levels, var_levels

    def plot_gps_shared_cbar(self, fig, axs, means, vars):
        """Plots mean and var for each expert in each output dim

        The rows iterate through experts and then output_dim
        e.g. row 1 = expert 1, output_dim 1
             row 2 = expert 1, output_dim 2
             row 3 = expert 2, output_dim 1
             row 4 = expert 2, output_dim 2

        :param axs: [num_experts*output_dim, 2]
        :param means: [num_data, output_dim, num_experts]
        :param vars: [num_data, output_dim, num_experts]
        """
        # output_dim = tf.shape(means)[1]
        output_dim = means.shape[1]
        row = 0
        mean_levels, var_levels = self.create_levels(means, vars)
        num_experts = means.shape[-1]
        for k in range(num_experts):
            for j in range(output_dim):
                if row != num_experts * output_dim - 1:
                    axs[row, 0].get_xaxis().set_visible(False)
                    axs[row, 1].get_xaxis().set_visible(False)
                mean_contf, var_contf = self.plot_gp_contf(
                    fig,
                    axs[row, :],
                    means[:, j, k],
                    vars[:, j, k],
                    mean_levels=mean_levels,
                    var_levels=var_levels,
                )
                row += 1
        mean_cbar = self.cbar(fig, axs[:, 0], mean_contf)
        var_cbar = self.cbar(fig, axs[:, 1], var_contf)
        return np.array([mean_cbar, var_cbar])

    def plot_experts_f(self, fig, axs):
        """Plots each experts latent function posterior in each output dim

        The rows iterate through experts and then output_dim
        e.g. row 1 = expert 1, output_dim 1
             row 2 = expert 1, output_dim 2
             row 3 = expert 2, output_dim 1
             row 4 = expert 2, output_dim 2

        :param axs: [num_experts*output_dim, 2]
        """
        tf.print("Plotting experts f...")
        means, vars = self.model.predict_experts_fs(self.test_inputs)
        # mean_levels = tf.linspace(
        #     tf.math.reduce_min(means), tf.math.reduce_max(means), self.num_levels
        # )
        # var_levels = tf.linspace(
        #     tf.math.reduce_min(vars), tf.math.reduce_max(vars), self.num_levels
        # )
        return self.plot_gps_shared_cbar(fig, axs, means, vars)
        # mean_levels, var_levels = self.create_levels(means, vars)
        # num_experts = means.shape[-1]
        # output_dim = means.shape[-2]
        # row = 0
        # cbars = []
        # for k in range(num_experts):
        #     for j in range(output_dim):
        #         if row != num_experts * output_dim - 1:
        #             axs[row, 0].get_xaxis().set_visible(False)
        #             axs[row, 1].get_xaxis().set_visible(False)
        #         cbars.append(
        #             self.plot_gp(fig, axs[row, :], means[:, j, k], vars[:, j, k])
        #         )
        #         row += 1
        # return np.array(cbars)

    def plot_experts_y(self, fig, axs):
        """Plots each experts predictive posterior in each output dim

        The rows iterate through experts and then output_dim
        e.g. row 1 = expert 1, output_dim 1
             row 2 = expert 1, output_dim 2
             row 3 = expert 2, output_dim 1
             row 4 = expert 2, output_dim 2

        :param axs: [num_experts*output_dim, 2]
        """
        tf.print("Plotting experts y...")
        dists = self.model.predict_experts_dists(self.test_inputs)
        means = dists.mean()
        vars = dists.variance()
        return self.plot_gps_shared_cbar(fig, axs, means, vars)

    def plot_gating_network(self, fig, axs):
        """Plots mixing probabilities for each expert in each output dim

        :param axs: [num_experts, output_dim]
        """
        tf.print("Plotting gating network mixing probabilities...")
        mixing_probs = self.model.predict_mixing_probs(self.test_inputs)
        for ax in axs:
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
        for k in range(self.num_experts):
            if k < self.num_experts - 1:
                axs[k].get_xaxis().set_visible(False)
            contf = axs[k].tricontourf(
                self.test_inputs[:, 0],
                self.test_inputs[:, 1],
                mixing_probs[:, k],
                100,
                levels=self.levels,
                cmap=self.cmap,
            )
        cbar = self.cbar(fig, axs, contf)
        return cbar
        # if self.output_dim > 1:
        #     cbars = []
        #     for k in range(self.num_experts):
        #         for j in range(self.output_dim):
        #             if k < self.num_experts - 1:
        #                 axs[k, j].get_xaxis().set_visible(False)
        #             contf = axs[k, j].tricontourf(self.test_inputs[:, 0],
        #                                           self.test_inputs[:, 1],
        #                                           mixing_probs[:, j, k],
        #                                           100,
        #                                           levels=self.levels,
        #                                           cmap=self.cmap)
        #             if k == 0:
        #                 cbars.append(self.cbar(fig, axs[:, j], contf))
        #     return cbars
        # else:
        #     for k in range(self.num_experts):
        #         if k < self.num_experts - 1:
        #             axs[k].get_xaxis().set_visible(False)
        #         contf = axs[k].tricontourf(self.test_inputs[:, 0],
        #                                    self.test_inputs[:, 1],
        #                                    mixing_probs[:, 0, k],
        #                                    100,
        #                                    levels=self.levels,
        #                                    cmap=self.cmap)
        #     cbar = self.cbar(fig, axs, contf)
        #     return cbar

    def plot_gating_gps(self, fig, axs):
        """Plots mean and var of gating network gp

        :param axs: if num_experts > 2: [num_experts, 2] else [1, 2]
        """
        tf.print("Plotting gating network gps...")
        # if self.num_experts > 2:
        #     return self.plot_gps_shared_cbar(fig, axs, means, vars)
        # else:
        #     return self.plot_gp(fig, axs, means[0, :, 0], vars[0, :, 0])
        cbars = []
        if self.num_experts > 2:
            means, vars = self.model.gating_network.predict_fs(self.test_inputs)
            # TODO haven't test the >2 expert case
            cbars.append(
                self.plot_gps_shared_cbar(
                    fig, axs, tf.expand_dims(means, -2), tf.expand_dims(vars, -2)
                )
            )
        else:
            means, vars = self.model.gating_network.predict_f(
                self.test_inputs, full_cov=False
            )
            cbars.append(self.plot_gp(fig, axs, means[:, 0], vars[:, 0]))
            for ax in axs:
                ax.scatter(
                    self.model.gating_network.inducing_variable.Z[:, 0],
                    self.model.gating_network.inducing_variable.Z[:, 1],
                    marker="x",
                    alpha=0.01,
                    color="k"
                    # self.X[:, 0], self.X[:, 1], marker="x", alpha=0.01, color="k"
                )
        return np.array(cbars)
        # output_dim = means.shape[1]
        # for i in range(self.output_dim):
        #     if self.num_experts > 2:
        #         # TODO haven't test the >2 expert case
        #         cbars.append(
        #             self.plot_gps_shared_cbar(fig,
        #                                       axs[i:i + self.num_experts, :],
        #                                       means, vars))
        #     else:
        #         # return self.plot_gp(fig, axs, means[0, :, 0], vars[0, :, 0])
        #         cbars.append(
        #             self.plot_gp(fig, axs[i, :], means[:, i, 0],
        #                          vars[:, i, 0]))
        #         for ax in axs[i, :]:
        #             ax.scatter(self.X[:, 0],
        #                        self.X[:, 1],
        #                        marker='x',
        #                        alpha=0.01,
        #                        color='k')
        # return np.array(cbars)

    def plot_y(self, fig, axs):
        """Plots mean and var of moment matched predictive posterior

        :param axs: [output_dim, 2]
        """
        tf.print("Plotting y (moment matched)...")
        y_dist = self.model.predict_y(self.test_inputs)
        means = y_dist.mean()
        vars = y_dist.variance()
        if self.output_dim > 1:
            # add num_experts of 1 for correct broadcasting in plot_gps_shared_cbar
            means = tf.expand_dims(means, -1)
            vars = tf.expand_dims(vars, -1)
            return self.plot_gps_shared_cbar(fig, axs, means, vars)
        else:
            return self.plot_gp(fig, axs, means[:, 0], vars[:, 0])

    def plot_model(self):
        nrows = self.num_experts * self.output_dim
        fig, ax = plt.subplots(self.num_experts, self.output_dim)
        self.plot_gating_network(fig, ax)
        if self.num_experts > 2:
            num_gating_gps = self.num_experts
        else:
            num_gating_gps = 1
        num_gating_gps *= self.output_dim
        fig, axs = plt.subplots(num_gating_gps, 2)
        self.plot_gating_gps(fig, axs)
        fig, axs = plt.subplots(nrows, 2, figsize=(10, 4))
        self.plot_experts_f(fig, axs)
        fig, axs = plt.subplots(nrows, 2, figsize=(10, 4))
        self.plot_experts_y(fig, axs)
        fig, axs = plt.subplots(self.output_dim, 2, figsize=(10, 4))
        self.plot_y(fig, axs)

    def tf_monitor_task_group(
        self, log_dir, slow_period=500, gating_network_only=False
    ):
        ncols = 2
        nrows_y = self.output_dim
        nrows_experts = self.num_experts * self.output_dim
        if self.num_experts > 2:
            # num_gating_gps = self.num_experts * self.output_dim
            num_gating_gps = self.num_experts
        else:
            # num_gating_gps = 1 * self.output_dim
            num_gating_gps = 1
        image_task_experts_f = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_experts_f,
            name="experts_latent_function_posterior",
            fig_kw={"figsize": (10, 4)},
            subplots_kw={"nrows": nrows_experts, "ncols": ncols},
        )
        image_task_experts_y = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_experts_y,
            name="experts_output_posterior",
            fig_kw={"figsize": (10, 4)},
            subplots_kw={"nrows": nrows_experts, "ncols": ncols},
        )
        image_task_gating_gps = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_gating_gps,
            name="gating_network_gps_posteriors",
            fig_kw={"figsize": (10, 2 * num_gating_gps)},
            subplots_kw={"nrows": num_gating_gps, "ncols": 2},
        )
        image_task_gating = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_gating_network,
            name="gating_network_mixing_probabilities",
            subplots_kw={
                "nrows": self.num_experts,
                "ncols": 1
                # 'ncols': self.output_dim
            },
        )
        image_task_y = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_y,
            name="predictive_posterior_moment_matched",
            fig_kw={"figsize": (10, 2)},
            subplots_kw={"nrows": nrows_y, "ncols": ncols},
        )
        # image_tasks = [
        #     image_task_experts_y, image_task_experts_f, image_task_gating
        # ]
        if gating_network_only:
            image_tasks = [
                image_task_gating,
                image_task_gating_gps,
            ]
        else:
            image_tasks = [
                image_task_experts_y,
                image_task_experts_f,
                image_task_gating,
                image_task_gating_gps,
                image_task_y
                # image_task_y
            ]
        # image_tasks = [
        #     image_task_experts_y, image_task_experts_f, image_task_y
        # ]
        return MonitorTaskGroup(image_tasks, period=slow_period)
