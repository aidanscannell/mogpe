#!/usr/bin/env python3
import pathlib
import os
import shutil
import pickle
from datetime import datetime

import gpflow as gpf
import numpy as np
import tensorflow as tf
from gpflow.monitor import ModelToTensorBoard, MonitorTaskGroup, ScalarToTensorBoard
from mogpe.training.metrics import (
    build_negative_log_predictive_density,
    build_mean_absolute_error,
    build_root_mean_squared_error,
)

# from .toml_config_parsers.model_parsers import create_mosvgpe_model_from_config
from .toml_config_parsers.model_parsers import MixtureOfSVGPExperts_from_toml


def update_model_from_checkpoint(model, ckpt_dir):
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)
    print("Restored Model")
    gpf.utilities.print_summary(model)
    return model


def load_model_from_config_and_checkpoint(config_file, ckpt_dir, dataset):
    model = MixtureOfSVGPExperts_from_toml(config_file, dataset)
    # model = create_mosvgpe_model_from_config(config_file, X)
    # print("Initial Model from config_file")
    # gpf.utilities.print_summary(model)
    return update_model_from_checkpoint(model, ckpt_dir)


def save_model(model, save_dir=None):
    if save_dir is None:
        save_dir = str(pathlib.Path(tempfile.gettempdir()))
    params = gpf.utilities.parameter_dict(model)
    gpf.utilities.multiple_assign(model, params)

    frozen_model = gpf.utilities.freeze(model)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)])
    def predict_experts_ys(Xnew):
        dists = frozen_model.predict_experts_dists(Xnew)
        return dists.mean(), dists.variance()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)])
    def predict_experts_fs(Xnew):
        return frozen_model.predict_experts_fs(Xnew)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)])
    def predict_gating_fs(Xnew):
        return frozen_model.gating_network.predict_fs(Xnew)

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)])
    # def predict_y(Xnew):
    #     dist = frozen_model.predict_y(Xnew)
    #     return dist

    # @tf.function(input_signature=[tf.TensorSpec(shape=[3, 2], dtype=tf.float64), tf.TensorSpec([], dtype=tf.int64)])
    # def predict_y_samples(Xnew, num_samples):
    #     # return frozen_model.predict_y_samples(Xnew, num_samples)
    #     return frozen_model.predict_y(Xnew).sample(num_samples)

    # TODO Needs to be fixed number of data points or ndiag_mc fails
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float64)])
    def predict_mixing_probs(Xnew):
        probs = frozen_model.predict_mixing_probs(Xnew, num_inducing_samples=None)
        return probs

    module_to_save = tf.Module()

    module_to_save.predict_experts_ys = predict_experts_ys
    module_to_save.predict_experts_fs = predict_experts_fs
    module_to_save.predict_gating_fs = predict_gating_fs
    # module_to_save.predict_y= predict_y
    # module_to_save.predict_y_samples = predict_y_samples
    module_to_save.predict_mixing_probs = predict_mixing_probs

    tf.saved_model.save(module_to_save, save_dir)


def save_models_param_dict(model, save_dir):
    save_model_dir = save_dir + "/param_dict.pickle"
    param_dict = gpf.utilities.parameter_dict(model)
    print("param dict")
    print(param_dict)
    f = open(save_model_dir, "wb")
    pickle.dump(param_dict, f)
    f.close()


def create_tf_dataset(dataset, num_data, batch_size):
    prefetch_size = tf.data.experimental.AUTOTUNE
    shuffle_buffer_size = num_data // 2
    num_batches_per_epoch = num_data // batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_dataset = (
        train_dataset.repeat()
        .prefetch(prefetch_size)
        .shuffle(buffer_size=shuffle_buffer_size)
        .batch(batch_size, drop_remainder=True)
    )
    return train_dataset, num_batches_per_epoch


def init_fast_tasks(log_dir, model=None, training_loss=None, fast_tasks_period=10):
    fast_tasks = []
    if training_loss is not None:
        fast_tasks.append(ScalarToTensorBoard(log_dir, training_loss, "training_loss"))
    if model is not None:
        fast_tasks.append(ModelToTensorBoard(log_dir, model))
    return MonitorTaskGroup(fast_tasks, period=fast_tasks_period)


def init_fast_tasks_bounds(
    log_dir,
    train_dataset,
    model,
    test_dataset=None,
    training_loss=None,
    fast_tasks_period=10,
):
    further_train_dataset_iter = iter(train_dataset)
    tight_train_dataset_iter = iter(train_dataset)

    @tf.function
    def elbo_further():
        batch = next(further_train_dataset_iter)
        return model.lower_bound_further(batch)

    @tf.function
    def elbo_tight():
        batch = next(tight_train_dataset_iter)
        return model.lower_bound_tight(batch)

    fast_tasks = []
    fast_tasks.append(ScalarToTensorBoard(log_dir, elbo_further, "elbo_further"))
    fast_tasks.append(ScalarToTensorBoard(log_dir, elbo_tight, "elbo_tight"))
    if training_loss is not None:
        fast_tasks.append(ScalarToTensorBoard(log_dir, training_loss, "training_loss"))
    if test_dataset is not None:
        further_test_dataset_iter = iter(test_dataset)
        tight_test_dataset_iter = iter(test_dataset)

        @tf.function
        def elbo_further_test():
            batch = next(further_test_dataset_iter)
            return model.lower_bound_further(batch)
            # return model.lower_bound_further(test_dataset)

        @tf.function
        def elbo_tight_test():
            batch = next(tight_test_dataset_iter)
            return model.lower_bound_tight(batch)
            # return model.lower_bound_tight(test_dataset)

        fast_tasks.append(
            ScalarToTensorBoard(
                log_dir,
                build_negative_log_predictive_density(model, iter(test_dataset)),
                "NLPD",
            )
        )
        fast_tasks.append(
            ScalarToTensorBoard(
                log_dir,
                build_mean_absolute_error(model, iter(test_dataset)),
                "MAE",
            )
        )
        fast_tasks.append(
            ScalarToTensorBoard(
                log_dir,
                build_root_mean_squared_error(model, iter(test_dataset)),
                "RMSE",
            )
        )
        fast_tasks.append(
            ScalarToTensorBoard(log_dir, elbo_further_test, "elbo_further_test")
        )
        fast_tasks.append(
            ScalarToTensorBoard(log_dir, elbo_tight_test, "elbo_tight_test")
        )
    fast_tasks.append(ModelToTensorBoard(log_dir, model))
    return MonitorTaskGroup(fast_tasks, period=fast_tasks_period)


def create_log_dir(
    log_dir,
    num_experts,
    batch_size,
    learning_rate=0.001,
    bound="further",
    num_inducing=None,
    config_file=None,
):
    log_dir = (
        log_dir
        + "/"
        + str(num_experts)
        + "_experts/batch_size_"
        + str(batch_size)
        + "/learning_rate_"
        + str(learning_rate)
        + "/"
        + bound
        + "_bound/"
    )
    if num_inducing is not None:
        log_dir = log_dir + "num_inducing_" + str(num_inducing) + "/"
    log_dir = log_dir + datetime.now().strftime("%m-%d-%H%M%S")
    os.makedirs(log_dir)
    if config_file is not None:
        try:
            shutil.copy(config_file, log_dir)
        except:
            print("Failed to copy config_file to log_dir")
    return log_dir


def init_inducing_variables(X, num_inducing):
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    input_dim = X.shape[1]
    idx = np.random.choice(range(X.shape[0]), size=num_inducing, replace=False)
    inducing_inputs = X[idx, :].reshape(num_inducing, input_dim)
    return gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(inducing_inputs)
    )


if __name__ == "__main__":

    config_file = "../../examples/mcycle/configs/config_2_experts.toml"
    ckpt_dir = "../../examples/logs/mcycle/two_experts/11-14-164351"

    model = load_model_from_config_and_checkpoint(config_file, ckpt_dir)
    gpf.utilities.print_summary(model)
    Xnew = np.linspace(0, 3, 100).reshape([100, 1])
    y_dist = model.predict_y(Xnew)
    ymu = y_dist.mean()
    import matplotlib.pyplot as plt

    plt.plot(Xnew, ymu)
    plt.show()
