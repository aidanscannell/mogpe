#!/usr/bin/env python3
from typing import List, Optional

from mogpe.custom_types import Dataset
from mogpe.keras.mixture_of_experts import MixtureOfSVGPExperts
from mogpe.keras.plotting import MixtureOfSVGPExpertsContourPlotter

from .tensorboard import PlotFn, TensorboardImageCallback


def build_contour_plotter_callbacks(
    mosvgpe: MixtureOfSVGPExperts,
    dataset: Dataset,
    logging_epoch_freq: Optional[int] = 10,
    log_dir: Optional[str] = "./logs",
) -> List[PlotFn]:
    mosvgpe_plotter = MixtureOfSVGPExpertsContourPlotter(mosvgpe, dataset=dataset)

    experts_plotting_cb = TensorboardImageCallback(
        plot_fn=mosvgpe_plotter.plot_experts_gps,
        logging_epoch_freq=logging_epoch_freq,
        log_dir=log_dir,
        name="Experts' latent function GPs",
    )
    gating_gps_plotting_cb = TensorboardImageCallback(
        plot_fn=mosvgpe_plotter.plot_gating_network_gps,
        logging_epoch_freq=logging_epoch_freq,
        log_dir=log_dir,
        name="Gating function GPs",
    )
    mixing_probs_plotting_cb = TensorboardImageCallback(
        plot_fn=mosvgpe_plotter.plot_mixing_probs,
        logging_epoch_freq=logging_epoch_freq,
        log_dir=log_dir,
        name="Mixing probabilities",
    )
    return [experts_plotting_cb, mixing_probs_plotting_cb, gating_gps_plotting_cb]
