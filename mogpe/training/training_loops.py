#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import pathlib
import time
import tensorflow as tf

from gpflow.monitor import ImageToTensorBoard, MonitorTaskGroup, Monitor


def training_tf_loop(model,
                     training_loss,
                     epochs: int = 1,
                     num_batches_per_epoch: int = 1,
                     logging_epoch_freq: int = 100):
    """Runs Adam optimizer on model with training_loss (no monitoring).

    :param model: The model to be trained.
    :param training_loss: A function that returns the training objective.
    :param epochs: The number of full data passes (epochs).
    :param num_batches_per_epoch: The number of batches per epoch
    :param logging_epoch_freq: The epoch frequency that the training loss is printed.
    """
    optimizer = tf.optimizers.Adam()

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
        # duration = t - time.time()
        # print("Iteration duration: ", duration)
        # t = time.time()


def monitored_training_tf_loop(model,
                               training_loss,
                               epochs: int = 1,
                               num_batches_per_epoch: int = 1,
                               fast_tasks: gpf.monitor.MonitorTaskGroup = None,
                               logging_epoch_freq: int = 100):
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
    optimizer = tf.optimizers.Adam()
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
            # duration = t - time.time()
            # print("Iteration duration: ", duration)
            # t = time.time()


def monitored_training_loop(model,
                            training_loss,
                            epochs: int = 1,
                            num_batches_per_epoch: int = 1,
                            fast_tasks: gpf.monitor.MonitorTaskGroup = None,
                            slow_tasks: gpf.monitor.MonitorTaskGroup = None,
                            logging_epoch_freq: int = 100,
                            save_dir: str=""):
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
    optimizer = tf.optimizers.Adam()
    # checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    @tf.function
    def tf_optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    monitor = Monitor(fast_tasks, slow_tasks)

    t = time.time()
    for epoch in range(epochs):
        for _ in range(num_batches_per_epoch):
            tf_optimization_step()
            # duration = t - time.time()
            # print("Iteration duration: ", duration)
            # t = time.time()
        monitor(epoch)
        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
            # save_model(model, save_model_dir)
        # duration = t - time.time()
        # print("Iteration duration: ", duration)
        # t = time.time()
