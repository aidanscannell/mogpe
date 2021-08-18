#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import tensorflow as tf
from config import config_from_toml
from mogpe.helpers.plotter import Plotter2D

# from mogpe.training import create_mosvgpe_model_from_config, monitored_training_loop
from mogpe.training import monitored_training_loop
from mogpe.training import MixtureOfSVGPExperts_from_toml
from mogpe.training.utils import create_log_dir, create_tf_dataset, init_fast_tasks
from quadcopter.load_data import load_quadcopter_dataset

tf.random.set_seed(42)
np.random.seed(42)


# Define input region (rectangle) to remove data from.
# This is done to test the models ability to capture epistemic unc.
x1_low = -1.0
x1_high = 1.0
x2_low = -1.0
x2_high = 3.0
# x1_low = 0.0
# x1_high = 1.0
# x2_low = 0.0
# x2_high = 3.0


if __name__ == "__main__":
    config_file = "./quadcopter/configs/config_2_experts.toml"  # path to model config
    config_file = "./quadcopter/configs/config_3_experts.toml"  # path to model config
    log_dir = "./logs/quadcopter"  # dir to store tensorboard logs

    # Set path to data set npz file
    # data_file = "./quadcopter/data/quadcopter_data_step_10_direction_down.npz"
    data_file = "./quadcopter/data/quadcopter_data_step_20_direction_down.npz"
    # data_file = "./quadcopter/data/quadcopter_data_step_40_direction_down.npz"

    # Load config (with model and training params) from toml file
    cfg = config_from_toml(config_file, read_from_file=True)

    # Load mcycle data set and create a tf dataset that can be iterated
    dataset = load_quadcopter_dataset(data_file)
    X, Y = dataset
    num_data = X.shape[0]
    train_dataset, num_batches_per_epoch = create_tf_dataset(
        dataset, num_data, cfg.batch_size
    )

    # Parse the toml config file to create MixtureOfSVGPExperts model
    # model = create_mosvgpe_model_from_config(config_file, X)
    model = MixtureOfSVGPExperts_from_toml(config_file, dataset)
    gpf.utilities.print_summary(model)

    # Build training loss closure
    training_loss = model.training_loss_closure(iter(train_dataset))

    # Create monitor tasks (plots/elbo/model params etc)
    log_dir = create_log_dir(log_dir, model.num_experts)
    plotter = Plotter2D(model, X, Y)
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
        fast_tasks=fast_tasks,
        slow_tasks=slow_tasks,
        num_batches_per_epoch=num_batches_per_epoch,
        logging_epoch_freq=cfg.logging_epoch_freq,
        manager=manager,
    )
