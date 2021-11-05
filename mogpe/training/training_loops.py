#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import pathlib
import time
import tensorflow as tf

from gpflow.monitor import ImageToTensorBoard, MonitorTaskGroup, Monitor
from mogpe.training.utils import save_models_param_dict


def training_tf_loop(
    model,
    training_loss,
    epochs: int = 1,
    num_batches_per_epoch: int = 1,
    learning_rate: float = 0.001,
    logging_epoch_freq: int = 100,
    manager: tf.train.CheckpointManager = None,
):
    """Runs Adam optimizer on model with training_loss (no monitoring).

    :param model: The model to be trained.
    :param training_loss: A function that returns the training objective.
    :param epochs: The number of full data passes (epochs).
    :param num_batches_per_epoch: The number of batches per epoch
    :param logging_epoch_freq: The epoch frequency that the training loss is printed.
    """
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def tf_optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    # t = time.time()
    for epoch in range(epochs):
        for _ in range(num_batches_per_epoch):
            tf_optimization_step()
            # tf_optimization_step(model, training_loss, optimizer)
        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
            if manager is not None:
                manager.save()
        # duration = t - time.time()
        # print("Iteration duration: ", duration)
        # t = time.time()


def monitored_training_tf_loop(
    model,
    training_loss,
    epochs: int = 1,
    num_batches_per_epoch: int = 1,
    learning_rate: float = 0.001,
    fast_tasks: gpf.monitor.MonitorTaskGroup = None,
    logging_epoch_freq: int = 100,
    manager: tf.train.CheckpointManager = None,
):
    """Monitors Adam optimizer on model with training_loss.

    Both training and monitoring are inside tf.function (no image monitoring).
    This method only monitors the fast tasks as matplotlib code cannot be built
    in a TF graph.

    :param model: The model to be trained.
    :param training_loss: A function that returns the training objective.
    :param epochs: The number of full data passes (epochs).
    :param num_batches_per_epoch: The number of batches per epoch
    :param fast_tasks: gpflow monitor fast tasks e.g.
        MonitorTaskGroup([ScalarToTensorBoard(log_dir, training_loss, "elbo")])
    :param logging_epoch_freq: The epoch frequency that the training loss is printed.
    """
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    monitor = Monitor(fast_tasks)

    @tf.function
    def monitored_tf_opt_step(epoch):
        optimizer.minimize(training_loss, model.trainable_variables)
        monitor(epoch)

    # t = time.time()
    epochs = tf.constant(epochs)  # needs to be tf.const
    for epoch in tf.range(epochs):
        for _ in range(num_batches_per_epoch):
            monitored_tf_opt_step(epoch)
        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
            if manager is not None:
                manager.save()
            # duration = t - time.time()
            # print("Iteration duration: ", duration)
            # t = time.time()


def monitored_training_loop(
    model,
    training_loss,
    epochs: int = 1,
    num_batches_per_epoch: int = 1,
    learning_rate: float = 0.001,
    fast_tasks: gpf.monitor.MonitorTaskGroup = None,
    slow_tasks: gpf.monitor.MonitorTaskGroup = None,
    logging_epoch_freq: int = 100,
    manager: tf.train.CheckpointManager = None,
):
    """Monitors (with images) Adam optimizer on model with training_loss.

    Monitoring is not inside tf.function so this method will be slower than
    monitored_training_tf_loop.

    :param model: The model to be trained.
    :param training_loss: A function that returns the training objective.
    :param epochs: The number of full data passes (epochs).
    :param num_batches_per_epoch: The number of batches per epoch
    :param fast_tasks: gpflow monitor fast tasks e.g.
        MonitorTaskGroup([ScalarToTensorBoard(log_dir, training_loss, "elbo")])
    :param slow_tasks: gpflow monitor slow tasks e.g. plotting images
    :param logging_epoch_freq: The epoch frequency that the training loss is printed.
    """
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def tf_optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    monitor = Monitor(fast_tasks, slow_tasks)

    # print("num_batches_per_epoch")
    # print(num_batches_per_epoch)
    t = time.time()
    for epoch in range(epochs):
        for _ in range(num_batches_per_epoch):
            tf_optimization_step()
        monitor(epoch)
        epoch_id = epoch + 1
        # print(f"Epoch {epoch_id}\n-------------------------------")
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
            if manager is not None:
                manager.save()
            #     # save_models_param_dict(model, save_dir)
            # duration = time.time() - t
            # tf.print("Iteration duration: ", duration)
            # t = time.time()
