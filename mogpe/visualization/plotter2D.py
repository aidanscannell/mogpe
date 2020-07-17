import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import palettable

from gpflow.monitor import ImageToTensorBoard, MonitorTaskGroup

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
                 cmap=palettable.scientific.sequential.Bilbao_15.mpl_colormap):
        super().__init__(model, X, Y, num_samples, params)
        self.cmap = cmap
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

    def plot_gp(self, fig, axs, mean, var):
        def plot_contour(fig, ax, x, y, z):
            contf = ax.tricontourf(x, y, z, 100, cmap=self.cmap)
            cbar = fig.colorbar(contf, ax=ax)

        plot_contour(fig, axs[0], self.test_inputs[:, 0],
                     self.test_inputs[:, 1], mean)
        plot_contour(fig, axs[1], self.test_inputs[:, 0],
                     self.test_inputs[:, 1], var)

    def plot_experts_f(self, fig, axs):
        tf.print("Plotting experts f...")
        row = 0
        for k, expert in enumerate(self.model.experts.experts_list):
            for j in range(self.output_dim):
                mean, var = expert.predict_f(self.test_inputs)
                self.plot_gp(fig, axs[row, :], mean[:, j], var[:, j])
                row += 1

    def plot_experts_y(self, fig, axs):
        tf.print("Plotting experts y...")
        dists = self.model.predict_experts_dists(self.test_inputs)
        mean = dists.mean()
        var = dists.variance()
        row = 0
        for k, expert in enumerate(self.model.experts.experts_list):
            for j in range(self.output_dim):
                mean, var = expert.predict_f(self.test_inputs)
                self.plot_gp(fig, axs[row, :],
                             dists.mean()[:, j, k],
                             dists.variance()[:, j, k])
                row += 1

    def plot_gating_network(self, fig, ax):
        tf.print("Plotting gating network...")
        mixing_probs = self.model.predict_mixing_probs(self.test_inputs)
        for k in range(self.num_experts):
            ax.plot(self.test_inputs, mixing_probs[:, k], label=str(k + 1))
        ax.legend()

    def plot_samples(self, fig, ax, input_broadcast, y_samples, color=color_3):
        ax.scatter(input_broadcast,
                   y_samples,
                   marker='.',
                   s=4.9,
                   color=color,
                   lw=0.4,
                   rasterized=True,
                   alpha=0.2)

    def plot_y(self, fig, axs):
        tf.print("Plotting y (moment matched)...")
        y_dist = self.model.predict_y(self.test_inputs)
        for j in range(self.output_dim):
            self.plot_gp(fig, axs[j, :],
                         y_dist.mean()[:, j],
                         y_dist.variance()[:, j])

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
        image_task_experts_f = ImageToTensorBoard(
            log_dir,
            self.plot_experts_f,
            name="experts_latent_function_posterior",
            fig_kw={'figsize': (10, 4)},
            subplots_kw={
                'nrows': nrows_experts,
                'ncols': ncols
            })
        image_task_experts_y = ImageToTensorBoard(
            log_dir,
            self.plot_experts_y,
            name="experts_output_posterior",
            fig_kw={'figsize': (10, 4)},
            subplots_kw={
                'nrows': nrows_experts,
                'ncols': ncols
            })
        image_task_gating = ImageToTensorBoard(
            log_dir,
            self.plot_gating_network,
            name="gating_network_mixing_probabilities",
        )
        image_task_y = ImageToTensorBoard(log_dir,
                                          self.plot_y,
                                          name="predictive_posterior",
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
