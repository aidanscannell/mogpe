#!/usr/bin/env python3
import io
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

PlotFn = Callable[[], matplotlib.figure.Figure]


class TensorboardImageCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        plot_fn: PlotFn,
        logging_epoch_freq: int = 10,
        log_dir: Optional[str] = "./logs",
        name: Optional[str] = "",
    ):
        self.plot_fn = plot_fn
        self.logging_epoch_freq = logging_epoch_freq
        self.name = name

        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.logging_epoch_freq == 0:
            # figure = self.plot_fn(self.model)
            figure = self.plot_fn()
            with self.file_writer.as_default():
                tf.summary.image(self.name, plot_to_image(figure), step=epoch)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
