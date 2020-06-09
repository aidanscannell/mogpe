import gpflow as gpf
import numpy as np
import pathlib
import tempfile
import tensorflow as tf


def run_adam(model, train_dataset, minibatch_size, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf


def custom_dir(dataset_name, config_dict):
    # setup custom logging location based on config
    if config_dict['num_samples_expert_expectation'] == "None":
        expert_expectation = 'analytic'
    else:
        expert_expectation = 'sample'
    if config_dict['add_date_to_logging_dir'] == "True":
        log_dir_date = True
    else:
        log_dir_date = False
    custom_dir = dataset_name + "/" + expert_expectation + "-f/batch_size-" + str(
        config_dict['batch_size']) + "/num_inducing-" + str(
            config_dict['gating']['num_inducing'])
    return custom_dir, log_dir_date


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
