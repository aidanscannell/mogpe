#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import tensorflow as tf
from config import config_from_toml
from gpflow.models.util import data_input_to_tensor
from gpflow.monitor import (
    Monitor,
    ImageToTensorBoard,
    ModelToTensorBoard,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

# from mogpe.training import create_mosvgpe_model_from_config, monitored_training_loop
from mogpe.training import monitored_training_loop
from mogpe.training.metrics import (
    build_mean_absolute_error,
    build_negative_log_predictive_density,
    build_root_mean_squared_error,
)
from mogpe.training.utils import create_log_dir, create_tf_dataset

from mcycle.data.load_data import load_mcycle_dataset

tf.random.set_seed(42)
np.random.seed(42)


def init_gp_from_config(config_file: str):
    cfg = config_from_toml(config_file, read_from_file=True)

    # Load mcycle data set
    train_dataset, test_dataset = load_mcycle_dataset(
        cfg.data_file,
        plot=cfg.plot_dataset,
        standardise=cfg.standardise,
        test_split_size=cfg.test_split_size,
    )
    X, Y = train_dataset
    num_data, input_dim = X.shape

    # Create SVGP model
    kernel = gpf.kernels.SquaredExponential()
    model = gpf.models.GPR(
        data=train_dataset, kernel=kernel, mean_function=gpf.mean_functions.Constant()
    )
    gpf.utilities.print_summary(model)
    return model


def train_gp_on_mcycle_given_config(
    config_file: str,
):
    cfg = config_from_toml(config_file, read_from_file=True)
    model = init_gp_from_config(config_file)

    # Load mcycle data set
    train_dataset, test_dataset = load_mcycle_dataset(
        cfg.data_file,
        plot=cfg.plot_dataset,
        standardise=cfg.standardise,
        test_split_size=cfg.test_split_size,
    )
    X, Y = train_dataset
    num_data, input_dim = X.shape

    # Create tf dataset
    train_dataset_tf = data_input_to_tensor(train_dataset)
    test_dataset_tf = data_input_to_tensor(test_dataset)

    # Create monitor tasks (plots/elbo/model params etc)
    log_dir = create_log_dir(
        cfg.log_dir,
        num_experts=1,
        bound="None",
        batch_size="na",
        # learning_rate=cfg.learning_rate,
        # num_inducing=cfg.num_inducing,
    )
    num_test = 100
    test_inputs = np.linspace(
        tf.reduce_min(X) * 1.2, tf.reduce_max(X) * 1.2, num_test
    ).reshape(num_test, input_dim)

    def plot_gp(fig, ax, mean, var):
        ax.scatter(X, Y, marker="x", color="k", alpha=0.4)
        ax.plot(test_inputs, mean, "C0", lw=2)
        ax.fill_between(
            test_inputs[:, 0],
            mean[:, 0] - 1.96 * np.sqrt(var)[:, 0],
            mean[:, 0] + 1.96 * np.sqrt(var)[:, 0],
            color="C0",
            alpha=0.2,
        )

    def plot_f(fig, ax):
        mean, var = model.predict_f(test_inputs)
        plot_gp(fig, ax, mean, var)

    def plot_y(fig, ax):
        mean, var = model.predict_y(test_inputs)
        plot_gp(fig, ax, mean, var)

    image_task_f = ImageToTensorBoard(
        log_dir,
        plot_f,
        name="latent_function_posterior",
    )
    image_task_y = ImageToTensorBoard(
        log_dir,
        plot_y,
        name="predictive_output_posterior",
    )
    slow_tasks = MonitorTaskGroup(
        [image_task_f, image_task_y], period=cfg.slow_tasks_period
    )
    loss_task = ScalarToTensorBoard(log_dir, model.training_loss, "training_loss")

    nlpd_task = ScalarToTensorBoard(
        log_dir,
        build_negative_log_predictive_density(model, iter(test_dataset_tf)),
        "NLPD",
    )
    mae_task = ScalarToTensorBoard(
        log_dir,
        build_mean_absolute_error(model, iter(test_dataset_tf)),
        "MAE",
    )
    rmse_task = ScalarToTensorBoard(
        log_dir,
        build_root_mean_squared_error(model, iter(test_dataset_tf)),
        "RMSE",
    )
    model_task = ModelToTensorBoard(log_dir, model)
    fast_tasks = MonitorTaskGroup(
        [loss_task, model_task],
        # [loss_task, model_task, nlpd_task, mae_task, rmse_task],
        period=cfg.fast_tasks_period,
    )

    # Init checkpoint manager for saving model during training
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=cfg.num_ckpts)

    optimiser = gpf.optimizers.Scipy()
    monitor = Monitor(fast_tasks, slow_tasks)

    def callback(step, variables, values):
        monitor(step)
        manager.save()

    # Main training loop
    optimisation_result = optimiser.minimize(
        model.training_loss,
        model.trainable_variables,
        step_callback=callback,
        options={
            "disp": True,
            "maxiter": cfg.epochs,
        },
    )

    # trained_model = monitored_training_loop(
    #     model,
    #     training_loss,
    #     epochs=cfg.epochs,
    #     learning_rate=cfg.learning_rate,
    #     fast_tasks=fast_tasks,
    #     slow_tasks=slow_tasks,
    #     num_batches_per_epoch=num_batches_per_epoch,
    #     logging_epoch_freq=cfg.logging_epoch_freq,
    #     manager=manager,
    # )


if __name__ == "__main__":
    # Load config (with model and training params) from toml file
    config_file = "./mcycle/configs/config_gp.toml"  # path to config

    train_gp_on_mcycle_given_config(config_file)
