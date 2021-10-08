#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import tensorflow as tf
from config import config_from_toml
from mogpe.helpers.plotter import Plotter2D

# from mogpe.training import create_mosvgpe_model_from_config, monitored_training_loop
from mogpe.training import MixtureOfSVGPExperts_from_toml, monitored_training_loop
from mogpe.training.utils import create_log_dir, create_tf_dataset, init_fast_tasks
from quadcopter.data.load_data import load_quadcopter_dataset

tf.random.set_seed(42)
np.random.seed(42)


def train_mogpe_on_quadcopter_given_config(
    config_file: str,
    # config_file: str, data_file: str, log_dir: str = "./logs/quadcopter"
):
    # config_file = "./quadcopter/configs/config_3_experts.toml"  # path to config
    cfg = config_from_toml(config_file, read_from_file=True)

    # Load quadcopter data set
    try:
        trim_coords = np.array(cfg.trim_coords)
    except AttributeError:
        trim_coords = None
    try:
        standardise = cfg.standardise
    except AttributeError:
        standardise = False
    try:
        plot = cfg.plot_dataset
    except AttributeError:
        plot = False
    dataset = load_quadcopter_dataset(
        cfg.data_file,
        trim_coords=trim_coords,
        num_outputs=2,
        plot=plot,
        standardise=standardise,
    )

    # Parse the toml config file to create MixtureOfSVGPExperts model
    model = MixtureOfSVGPExperts_from_toml(config_file, dataset)
    gpf.utilities.print_summary(model)

    # Create tf dataset that can be iterated and build training loss closure
    train_dataset, num_batches_per_epoch = create_tf_dataset(
        dataset, num_data=dataset[0].shape[0], batch_size=cfg.batch_size
    )
    training_loss = model.training_loss_closure(iter(train_dataset))

    # Create monitor tasks (plots/elbo/model params etc)
    log_dir = create_log_dir(
        cfg.log_dir, model.num_experts, cfg.batch_size, learning_rate=cfg.learning_rate
    )
    plotter = Plotter2D(model, X=dataset[0], Y=dataset[1])
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
    config_file = "./quadcopter/configs/config_2_experts_subset.toml"  # path to config
    config_file = "./quadcopter/configs/config_2_experts_full.toml"  # path to config

    train_mogpe_on_quadcopter_given_config(config_file=config_file)
