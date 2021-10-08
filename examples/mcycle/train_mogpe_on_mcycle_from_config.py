#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import tensorflow as tf
from config import config_from_toml
from mogpe.helpers.plotter import Plotter1D

# from mogpe.training import create_mosvgpe_model_from_config, monitored_training_loop
from mogpe.training import MixtureOfSVGPExperts_from_toml, monitored_training_loop
from mogpe.training.utils import create_log_dir, create_tf_dataset, init_fast_tasks
from mcycle.data.load_data import load_mcycle_dataset

tf.random.set_seed(42)
np.random.seed(42)


def train_mogpe_on_mcycle_given_config(
    config_file: str,
):
    cfg = config_from_toml(config_file, read_from_file=True)

    # Load mcycle data set
    try:
        standardise = cfg.standardise
    except AttributeError:
        standardise = False
    try:
        plot = cfg.plot_dataset
    except AttributeError:
        plot = False
    dataset = load_mcycle_dataset(cfg.data_file, plot=plot, standardise=standardise)

    # Parse the toml config file to create MixtureOfSVGPExperts model
    model = MixtureOfSVGPExperts_from_toml(config_file, dataset)
    gpf.utilities.print_summary(model)
    gpf.set_trainable(model.gating_network.inducing_variable, False)
    for expert in model.experts.experts_list:
        gpf.set_trainable(expert.inducing_variable, False)
    gpf.utilities.print_summary(model)

    # Create tf dataset that can be iterated and build training loss closure
    train_dataset, num_batches_per_epoch = create_tf_dataset(
        dataset, num_data=dataset[0].shape[0], batch_size=cfg.batch_size
    )
    training_loss = model.training_loss_closure(iter(train_dataset))

    # Create monitor tasks (plots/elbo/model params etc)
    log_dir = create_log_dir(
        cfg.log_dir,
        model.num_experts,
        cfg.batch_size,
        learning_rate=cfg.learning_rate,
        bound=cfg.bound,
        num_inducing=cfg.experts[0]["inducing_points"]["num_inducing"],
    )
    plotter = Plotter1D(model, X=dataset[0], Y=dataset[1])
    slow_tasks = plotter.tf_monitor_task_group(log_dir, cfg.slow_tasks_period)
    fast_tasks = init_fast_tasks(log_dir, model, training_loss, cfg.fast_tasks_period)

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
    config_file = "./mcycle/configs/config_2_experts_subset.toml"  # path to config
    # config_file = "./mcycle/configs/config_3_experts_subset.toml"  # path to config

    train_mogpe_on_mcycle_given_config(config_file=config_file)
