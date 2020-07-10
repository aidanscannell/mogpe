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
                            logging_epoch_freq: int = 100):
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

    @tf.function
    def tf_optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    monitor = Monitor(fast_tasks, slow_tasks)

    # t = time.time()
    for epoch in range(epochs):
        for _ in range(num_batches_per_epoch):
            tf_optimization_step()
        monitor(epoch)
        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
        # duration = t - time.time()
        # print("Iteration duration: ", duration)
        # t = time.time()


def init_slow_tasks(plotter, num_experts, log_dir, slow_period=500):
    # Set up slow tasks
    image_task_experts_f = ImageToTensorBoard(
        log_dir,
        plotter.plot_experts_f,
        name="experts_latent_function_posterior",
        fig_kw={'figsize': (10, 4)},
        subplots_kw={
            'nrows': 1,
            'ncols': num_experts
        })
    image_task_experts_y = ImageToTensorBoard(log_dir,
                                              plotter.plot_experts_y,
                                              name="experts_output_posterior",
                                              fig_kw={'figsize': (10, 4)},
                                              subplots_kw={
                                                  'nrows': 1,
                                                  'ncols': num_experts
                                              })
    image_task_gating = ImageToTensorBoard(
        log_dir,
        plotter.plot_gating_network,
        name="gating_network_mixing_probabilities",
    )
    image_task_y = ImageToTensorBoard(
        log_dir,
        plotter.plot_y,
        name="predictive_posterior",
    )
    # image_tasks = [
    #     image_task_experts_y, image_task_experts_f, image_task_gating
    # ]
    image_tasks = [
        image_task_experts_y, image_task_experts_f, image_task_gating,
        image_task_y
    ]
    return MonitorTaskGroup(image_tasks, period=slow_period)


def save_model(model, save_dir=None):
    if save_dir is None:
        save_dir = str(pathlib.Path(tempfile.gettempdir()))
    params = gpf.utilities.parameter_dict(model)
    gpf.utilities.multiple_assign(model, params)

    frozen_model = gpf.utilities.freeze(model)

    module_to_save = tf.Module()
    predict_y_mm_fn = tf.function(
        frozen_model.predict_y_moment_matched,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    sample_y_fn = tf.function(
        frozen_model.sample_y,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    predict_experts_fs_fn = tf.function(
        frozen_model.predict_experts_fs,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    predict_experts_ys_fn = tf.function(
        frozen_model.predict_experts_ys,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    predict_gating_h_fn = tf.function(
        frozen_model.predict_gating_h,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    predict_mixing_probs_fn = tf.function(
        frozen_model.predict_mixing_probs,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    module_to_save.predict_y_moment_matched = predict_y_mm_fn
    module_to_save.sample_y = sample_y_fn
    module_to_save.predict_experts_fs = predict_experts_fs_fn
    module_to_save.predict_experts_ys = predict_experts_ys_fn
    module_to_save.predict_gating_h = predict_gating_h_fn
    module_to_save.predict_mixing_probs = predict_mixing_probs_fn

    tf.saved_model.save(module_to_save, save_dir)