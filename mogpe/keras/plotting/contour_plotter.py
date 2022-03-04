#!/usr/bin/env python3
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import palettable
import tensorflow as tf
from mogpe.custom_types import Dataset, InputData
from mogpe.keras.mixture_of_experts import MixtureOfSVGPExperts
from mpl_toolkits.axes_grid1 import make_axes_locatable

CMAP = palettable.scientific.sequential.Bilbao_15.mpl_colormap


class MixtureOfSVGPExpertsContourPlotter:
    """Used to plot first two input dimensions using contour plots

    Can handle arbitrary number of experts and output dimensions.
    """

    def __init__(
        self,
        model: MixtureOfSVGPExperts,
        dataset: Dataset = None,
        test_inputs: Optional[InputData] = None,
        figsize: Tuple[float, float] = (12, 4),
        cmap=CMAP,
        static: bool = False,
    ):
        self.model = model
        self.num_experts = model.num_experts
        self.output_dim = model.experts_list[0].gp.num_latent_gps
        self.figsize = figsize
        self.cmap = cmap
        self.static = static

        if test_inputs is not None:
            self.test_inputs = test_inputs
        else:
            self.test_inputs = create_test_inputs(X=dataset[0])

        if self.static:
            self.h_means, self.h_vars = self.model.gating_network.predict_h(
                self.test_inputs
            )
            self.mixing_probs = self.model.gating_network.predict_mixing_probs(
                self.test_inputs
            )
            self.f_means, self.f_vars = self.model.predict_experts_f(self.test_inputs)

    def plot_model(self):
        self.plot_gating_network_gps()
        self.plot_mixing_probs()
        self.plot_experts_gps()

    def plot_gating_network_gps(self):
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * self.num_experts))
        gs = fig.add_gridspec(self.num_experts, 2)
        axs = gs.subplots(sharex=True, sharey=True)
        # fig.suptitle("Gating GPs")
        if not self.static:
            self.h_means, self.h_vars = self.model.gating_network.predict_h(
                self.test_inputs
            )
        for k in range(self.num_experts):
            self.plot_gp(
                axs[k, :],
                self.h_means[:, k],
                self.h_vars[:, k],
                label="Gating function $h_" + str(k + 1) + "(\mathbf{x})$",
            )
        [ax.set_ylabel("") for ax in axs[:, 1:].flat]
        [ax.set_xlabel("") for ax in axs[:-1, :].flat]
        fig.tight_layout()
        return fig

    def plot_mixing_probs(self):
        fig = plt.figure(figsize=self.figsize)
        # fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] / 2))
        # fig.suptitle("Mixing probabilities")
        gs = fig.add_gridspec(1, self.num_experts)
        axs = gs.subplots(sharex=True, sharey=True)
        if not self.static:
            self.mixing_probs = self.model.gating_network.predict_mixing_probs(
                self.test_inputs
            )
        for k in range(self.num_experts):
            self.plot_contf(axs[k], self.mixing_probs[:, k])
            axs[k].set_title("$\Pr(\\alpha=" + str(k + 1) + " \mid \mathbf{x})$")
        [ax.set_ylabel("") for ax in axs[1:].flat]
        fig.tight_layout()
        return fig

    def plot_experts_gps(self):
        nrows = self.num_experts * self.output_dim
        # fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * nrows / 2))
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * nrows))
        # fig.suptitle("Experts' latent function GPs")
        gs = fig.add_gridspec(nrows, 2)
        axs = gs.subplots(sharex=True, sharey=True)
        row = 0
        if not self.static:
            self.f_means, self.f_vars = self.model.predict_experts_f(self.test_inputs)
        for k in range(self.model.num_experts):
            for j in range(self.output_dim):
                self.plot_gp(
                    axs[row, :],
                    self.f_means[:, j, k],
                    self.f_vars[:, j, k],
                    label="Expert {} (output {})".format(k + 1, j + 1),
                )
                row += 1
        [ax.set_ylabel("") for ax in axs[:, 1:].flat]
        [ax.set_xlabel("") for ax in axs[:-1, :].flat]
        fig.tight_layout()
        return fig

    def plot_gp(self, axs, mean, var, label: str = ""):
        axs[0].set_title(label + " mean")
        axs[1].set_title(label + " variance")
        return self.plot_contf(axs[0], mean), self.plot_contf(axs[1], var)

    def plot_contf(self, ax, z):
        contf = ax.tricontourf(
            self.test_inputs[:, 0], self.test_inputs[:, 1], z, levels=10, cmap=self.cmap
        )
        plt.colorbar(contf, ax=ax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        return contf

    # def add_cbar(self, fig, ax, contfs, label=""):
    #     # divider = make_axes_locatable(ax)

    #     if isinstance(ax, np.ndarray):
    #         divider = make_axes_locatable(ax[0])
    #     else:
    #         divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("top", size="5%", pad=0.05)

    #     cbar = fig.colorbar(
    #         contfs, use_gridspec=True, cax=cax, orientation="horizontal"
    #     )
    #     cax.xaxis.set_ticks_position("top")
    #     cax.xaxis.set_label_position("top")
    #     cbar.set_label(label)
    #     return cbar


def create_test_inputs(X, num_test: int = 400):
    num_test = 400
    num_test = 1600
    factor = 1.2
    sqrtN = int(np.sqrt(num_test))
    xx = np.linspace(tf.reduce_min(X[:, 0]) * factor, np.max(X[:, 0]) * factor, sqrtN)
    yy = np.linspace(tf.reduce_min(X[:, 1]) * factor, np.max(X[:, 1]) * factor, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    if X.shape[-1] > 2:
        zeros = np.zeros((num_test, X.shape[-1] - 2))
        test_inputs = np.concatenate([test_inputs, zeros], -1)
    return test_inputs
