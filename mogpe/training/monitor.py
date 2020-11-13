#!/usr/bin/env python3
""" Tasks that write to TensorBoard """

from io import BytesIO
from typing import Any, Callable, Dict, Optional

import numpy as np
import tensorflow as tf

from gpflow.monitor import ToTensorBoard


class ImageWithCbarToTensorBoard(ToTensorBoard):
    def __init__(
        self,
        log_dir: str,
        plotting_function: Callable[
            ["matplotlib.figure.Figure", "matplotlib.figure.Axes"], "matplotlib.figure.Figure"
        ],
        name: Optional[str] = None,
        *,
        fig_kw: Optional[Dict[str, Any]] = None,
        subplots_kw: Optional[Dict[str, Any]] = None,
        # colorbar_kw: Optional[Dict[str, Any]] = None,
    ):
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be a nested: for example, './logs/my_run/'.
        :param plotting_function: function performing the plotting.
        :param name: name used in TensorBoard.
        :params fig_kw: Keywords to be passed to Figure constructor, such as `figsize`.
        :params subplots_kw: Keywords to be passed to figure.subplots constructor, such as
            `nrows`, `ncols`, `sharex`, `sharey`. By default the default values
            from matplotlib.pyplot as used.
        """
        super().__init__(log_dir)
        self.plotting_function = plotting_function
        self.name = name
        # self.file_writer = tf.summary.create_file_writer(log_dir)
        self.name = name
        self.fig_kw = fig_kw or {}
        self.subplots_kw = subplots_kw or {}
        self.cbs = None
        # self.colorbar_kw = colorbar_kw or {}

        # self.fig = Figure(**self.fig_kw)
        # if self.subplots_kw != {}:
        #     self.axes = self.fig.subplots(**self.subplots_kw)
        # else:
        #     self.axes = self.fig.add_subplot(111)
        # # if self.colorbar_kw != {}:
        # #     self.cbs = self.fig.colorbar(**self.subplots_kw)

        try:
            from matplotlib.figure import Figure
        except ImportError:
            raise RuntimeError("ImageWithCbarToTensorBoard requires the matplotlib package to be installed")

        self.fig = Figure(**self.fig_kw)
        if self.subplots_kw != {}:
            self.axes = self.fig.subplots(**self.subplots_kw)
        else:
            self.axes = self.fig.add_subplot(111)

    def _clear_axes(self):
        if isinstance(self.axes, np.ndarray):
            for ax in self.axes.flatten():
                ax.clear()
        else:
            self.axes.clear()

    def _clear_cbs(self):
        # tf.print('inside clear cbs')
        if isinstance(self.cbs, np.ndarray):
            # tf.print('cbs is ndarray')
            for cb in self.cbs.flatten():
                # cb.ax.clear()
                cb.remove()
        elif isinstance(self.cbs, list):
            # tf.print('cbs is List')
            for cb in self.cbs:
                cb.remove()
        else:
            # tf.print('cbs isnt ndarray or List')
            # self.cbs.ax.clear()
            self.cbs.remove()

    def run(self, **unused_kwargs):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self._clear_axes()
        # if self.colorbar_kw != {}:
        if self.cbs is not None:
            # tf.print('cbs is not None')
            self._clear_cbs()
        self.cbs = self.plotting_function(self.fig, self.axes)
        # else:
        #     self.plotting_function(self.fig, self.axes)
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()

        # get PNG data from the figure
        png_buffer = BytesIO()
        canvas.print_png(png_buffer)
        png_encoded = png_buffer.getvalue()
        png_buffer.close()

        image_tensor = tf.io.decode_png(png_encoded)[None]

        # Write to TensorBoard
        tf.summary.image(self.name, image_tensor, step=self.current_step)
