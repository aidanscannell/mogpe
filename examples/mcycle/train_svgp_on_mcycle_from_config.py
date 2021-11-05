#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import tensorflow as tf
from config import config_from_toml
from gpflow.monitor import (
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


def init_svgp_from_config(config_file: str):
    cfg = config_from_toml(config_file, read_from_file=True)

    # Load mcycle data set
    # X, Y = load_mcycle_dataset(
    #     cfg.data_file, plot=cfg.plot_dataset, standardise=cfg.standardise
    # )
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
    idx = np.random.choice(range(X.shape[0]), size=cfg.num_inducing, replace=False)
    inducing_inputs = X.numpy()[idx, :].reshape(cfg.num_inducing, input_dim)
    inducing_variable = gpf.inducing_variables.InducingPoints(inducing_inputs)
    model = gpf.models.SVGP(
        kernel,
        gpf.likelihoods.Gaussian(),
        inducing_variable,
        num_data=num_data,
        mean_function=gpf.mean_functions.Constant(),
    )
    gpf.utilities.print_summary(model)
    return model


def train_svgp_on_mcycle_given_config(
    config_file: str,
):
    cfg = config_from_toml(config_file, read_from_file=True)
    model = init_svgp_from_config(config_file)

    # Load mcycle data set
    # X, Y = load_mcycle_dataset(
    #     cfg.data_file, plot=cfg.plot_dataset, standardise=cfg.standardise
    # )
    train_dataset, test_dataset = load_mcycle_dataset(
        cfg.data_file,
        plot=cfg.plot_dataset,
        standardise=cfg.standardise,
        test_split_size=cfg.test_split_size,
    )
    X, Y = train_dataset
    num_data, input_dim = X.shape

    # Create tf dataset that can be iterated and build training loss closure
    train_dataset_tf, num_batches_per_epoch = create_tf_dataset(
        dataset=train_dataset, num_data=num_data, batch_size=cfg.batch_size
    )
    if test_dataset is not None:
        test_dataset_tf, _ = create_tf_dataset(
            dataset=test_dataset, num_data=num_data, batch_size=cfg.batch_size
        )
    else:
        test_dataset_tf = None
    training_loss = model.training_loss_closure(iter(train_dataset_tf))

    # Create monitor tasks (plots/elbo/model params etc)
    log_dir = create_log_dir(
        cfg.log_dir,
        num_experts=1,
        bound="normal",
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_inducing=cfg.num_inducing,
        config_file=config_file,
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
    loss_task = ScalarToTensorBoard(log_dir, training_loss, "training_loss")
    model_task = ModelToTensorBoard(log_dir, model)
    if test_dataset_tf is not None:
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
        fast_tasks = MonitorTaskGroup(
            [loss_task, model_task, nlpd_task, mae_task, rmse_task],
            period=cfg.fast_tasks_period,
        )
    else:
        fast_tasks = MonitorTaskGroup(
            [loss_task, model_task],
            period=cfg.fast_tasks_period,
        )

    # Init checkpoint manager for saving model during training
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=cfg.num_ckpts)

    # Main training loop
    trained_model = monitored_training_loop(
        model,
        training_loss,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        fast_tasks=fast_tasks,
        slow_tasks=slow_tasks,
        num_batches_per_epoch=num_batches_per_epoch,
        logging_epoch_freq=cfg.logging_epoch_freq,
        manager=manager,
    )


if __name__ == "__main__":
    # Load config (with model and training params) from toml file
    # config_file = "./mcycle/configs/config_svgp_m_16.toml"  # path to config
    # config_file = "./mcycle/configs/config_svgp_m_32.toml"  # path to config
    config_file = "./mcycle/configs/config_svgp_m_32_full.toml"  # path to config

    train_svgp_on_mcycle_given_config(config_file)
