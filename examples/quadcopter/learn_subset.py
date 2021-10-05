#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from quadcopter.train_model_from_config import train_model_from_config

tf.random.set_seed(42)
np.random.seed(42)


if __name__ == "__main__":
    # log_dir = "./logs/quadcopter/subset"  # dir to store tensorboard logs

    # Load config (with model and training params) from toml file
    config_file = "./quadcopter/configs/config_2_experts_subset.toml"  # path to config

    # data_file = "./quadcopter/data/quadcopter_data_step_10_direction_down.npz"
    # data_file = "./quadcopter/data/quadcopter_data_step_20_direction_down.npz"
    # data_file = "./quadcopter/data/quadcopter_data_step_40_direction_down.npz"

    train_model_from_config(config_file=config_file)
