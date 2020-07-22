import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import palettable

from gpflow.monitor import ImageToTensorBoard, MonitorTaskGroup, ImageColorBarToTensorBoard
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mogpe.visualization.plotter import Plotter

color_1 = 'olive'
color_2 = 'darkmagenta'
color_2 = 'darkslategrey'
color_3 = 'darkred'
color_3 = 'lime'
color_obs = 'red'


class Plotter2D(Plotter):
    def __init__(self,
                 model,
                 X,
                 Y,
                 test_inputs=None,
                 num_samples=100,
                 params=None,
                 num_levels=10,
                 cmap=palettable.scientific.sequential.Bilbao_15.mpl_colormap):
        super().__init__(model, X, Y, num_samples, params)
        self.cmap = cmap
        self.num_levels = num_levels
        if test_inputs is None:
            num_test = 400
            factor = 1.2
            sqrtN = int(np.sqrt(num_test))
            xx = np.linspace(
                tf.reduce_min(X[:, 0]) * factor,
                tf.reduce_max(X[:, 0]) * factor, sqrtN)
            yy = np.linspace(
                tf.reduce_min(X[:, 1]) * factor,
                tf.reduce_max(X[:, 1]) * factor, sqrtN)
            xx, yy = np.meshgrid(xx, yy)
            self.test_inputs = np.column_stack(
                [xx.reshape(-1), yy.reshape(-1)])
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
        mean_contf, var_contf = self.plot_gp_contf(fig, axs, mean, var,
                                                   mean_levels, var_levels)
        mean_cbar = self.cbar(fig, axs[0], mean_contf)
        var_cbar = self.cbar(fig, axs[1], var_contf)
        return np.array([mean_cbar, var_cbar])

    def plot_gp_contf(self,
                      fig,
                      axs,
                      mean,
                      var,
                      mean_levels=None,
                      var_levels=None):
        """Plots contours for mean and var side by side

        :param axs: [2,]
        :param mean: [num_data, 1]
        :param var: [num_data, 1]
        :param mean_levels: levels for mean contourf e.g. np.linspace(0, 1, 10)
        :param var_levels: levels for var contourf e.g. np.linspace(0, 1, 10)
        """
        mean_contf = axs[0].tricontourf(self.test_inputs[:, 0],
                                        self.test_inputs[:, 1],
                                        mean,
                                        100,
                                        levels=mean_levels,
                                        cmap=self.cmap)
        var_contf = axs[1].tricontourf(self.test_inputs[:, 0],
                                       self.test_inputs[:, 1],
                                       var,
                                       100,
                                       levels=var_levels,
                                       cmap=self.cmap)
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
        cbar = fig.colorbar(contf,
                            ax=ax,
                            use_gridspec=True,
                            cax=cax,
                            format='%0.2f',
                            orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        return cbar

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
        mean_levels = tf.linspace(tf.math.reduce_min(means),
                                  tf.math.reduce_max(means), self.num_levels)
        var_levels = tf.linspace(tf.math.reduce_min(vars),
                                 tf.math.reduce_max(vars), self.num_levels)
        row = 0
        num_experts = means.shape[-1]
        for k in range(num_experts):
            for j in range(self.output_dim):
                if row != num_experts * self.output_dim - 1:
                    axs[row, 0].get_xaxis().set_visible(False)
                    axs[row, 1].get_xaxis().set_visible(False)
                mean_contf, var_contf = self.plot_gp_contf(
                    fig,
                    axs[row, :],
                    means[:, j, k],
                    vars[:, j, k],
                    mean_levels=mean_levels,
                    var_levels=var_levels)
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
        mean_levels = tf.linspace(tf.math.reduce_min(means),
                                  tf.math.reduce_max(means), self.num_levels)
        var_levels = tf.linspace(tf.math.reduce_min(vars),
                                 tf.math.reduce_max(vars), self.num_levels)
        return self.plot_gps_shared_cbar(fig, axs, means, vars)

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
        mean_levels = tf.linspace(tf.math.reduce_min(means),
                                  tf.math.reduce_max(means), self.num_levels)
        var_levels = tf.linspace(tf.math.reduce_min(vars),
                                 tf.math.reduce_max(vars), self.num_levels)
        return self.plot_gps_shared_cbar(fig, axs, means, vars)

    def plot_gating_network(self, fig, axs):
        """Plots mixing probabilities for each expert in each output dim

        :param axs: [num_experts, output_dim]
        """
        tf.print("Plotting gating network...")
        mixing_probs = self.model.predict_mixing_probs(self.test_inputs)
        levels = np.linspace(0., 1., 5)
        print(mixing_probs.shape)
        if self.output_dim > 1:
            cbars = []
            for k in range(self.num_experts):
                for j in range(self.output_dim):
                    if k < self.num_experts - 1:
                        axs[k, j].get_xaxis().set_visible(False)
                    contf = axs[k, j].tricontourf(self.test_inputs[:, 0],
                                                  self.test_inputs[:, 1],
                                                  mixing_probs[:, j, k],
                                                  100,
                                                  levels=levels,
                                                  cmap=self.cmap)
                    if k == 0:
                        cbars.append(self.cbar(fig, axs[:, j], contf))
            return cbars
        else:
            for k in range(self.num_experts):
                if k < self.num_experts - 1:
                    axs[k].get_xaxis().set_visible(False)
                contf = axs[k].tricontourf(self.test_inputs[:, 0],
                                           self.test_inputs[:, 1],
                                           mixing_probs[:, 0, k],
                                           100,
                                           levels=levels,
                                           cmap=self.cmap)
            cbar = self.cbar(fig, axs, contf)
            return cbar

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
            mean_levels = tf.linspace(tf.math.reduce_min(means),
                                      tf.math.reduce_max(means),
                                      self.num_levels)
            var_levels = tf.linspace(tf.math.reduce_min(vars),
                                     tf.math.reduce_max(vars), self.num_levels)
            return self.plot_gps_shared_cbar(fig, axs, means, vars)
        else:
            return self.plot_gp(fig, axs, means[:, 0], vars[:, 0])

    def plot_model(self):
        nrows = self.num_experts * self.output_dim
        fig, ax = plt.subplots(1, 1)
        self.plot_gating_network(fig, ax)
        fig, axs = plt.subplots(nrows, 2, figsize=(10, 4))
        self.plot_experts_f(fig, axs)
        fig, axs = plt.subplots(nrows, 2, figsize=(10, 4))
        self.plot_experts_y(fig, axs)
        fig, axs = plt.subplots(self.output_dim, 2, figsize=(10, 4))
        self.plot_y(fig, axs)

    def tf_monitor_task_group(self, log_dir, slow_period=500):
        ncols = 2
        nrows_y = self.output_dim
        nrows_experts = self.num_experts * self.output_dim
        image_task_experts_f = ImageColorBarToTensorBoard(
            log_dir,
            self.plot_experts_f,
            name="experts_latent_function_posterior",
            fig_kw={'figsize': (10, 4)},
            subplots_kw={
                'nrows': nrows_experts,
                'ncols': ncols
            })
        image_task_experts_y = ImageColorBarToTensorBoard(
            log_dir,
            self.plot_experts_y,
            name="experts_output_posterior",
            fig_kw={'figsize': (10, 4)},
            subplots_kw={
                'nrows': nrows_experts,
                'ncols': ncols
            })
        image_task_gating = ImageColorBarToTensorBoard(
            log_dir,
            self.plot_gating_network,
            name="gating_network_mixing_probabilities",
            subplots_kw={
                'nrows': self.num_experts,
                'ncols': self.output_dim
            })
        image_task_y = ImageColorBarToTensorBoard(
            log_dir,
            self.plot_y,
            name="predictive_posterior_moment_matched",
            fig_kw={'figsize': (10, 2)},
            subplots_kw={
                'nrows': nrows_y,
                'ncols': ncols
            })
        # image_tasks = [
        #     image_task_experts_y, image_task_experts_f, image_task_gating
        # ]
        image_tasks = [
            image_task_experts_y, image_task_experts_f, image_task_gating,
            image_task_y
        ]
        # image_tasks = [
        #     image_task_experts_y, image_task_experts_f, image_task_y
        # ]
        return MonitorTaskGroup(image_tasks, period=slow_period)


if __name__ == "__main__":
    import json
    from bunch import Bunch
    from mogpe.data.utils import load_quadcopter_dataset
    from mogpe.training.parse_config import parse_model_from_config_file

    data_file = '../../data/processed/quadcopter_turbulence.npz'
    data = load_quadcopter_dataset(filename=data_file, standardise=False)
    X, Y = data

    config_file = '../../configs/quadcopter.json'
    model = parse_model_from_config_file(config_file)

    plotter = Plotter2D(model, X=X, Y=Y)
    plotter.plot_model()
    # plotter.plot_experts()
    # plotter.plot_gating_netowrk()
    plt.show()
