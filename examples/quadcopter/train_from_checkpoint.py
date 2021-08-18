#!/usr/bin/env python3
from mogpe.training import (
    train_from_config_and_checkpoint,
    train_from_config_and_dataset,
)

from quadcopter.load_data import load_quadcopter_dataset


if __name__ == "__main__":
    # Set path to data set npz file
    # data_file = "./data/quadcopter_data.npz"
    data_file = "./quadcopter/data/quadcopter_data_step_20.npz"
    data_file = "./quadcopter/data/quadcopter_data_step_40.npz"
    data_file = "./quadcopter/data/quadcopter_data.npz"

    # Set path to training config
    config_file = "./quadcopter/configs/config_2_experts.toml"
    ckpt_dir = "./logs/quadcopter/two_experts/08-13-172243"
    ckpt_dir = "./logs/quadcopter/two_experts/08-13-190426"

    # Load mcycle data set
    dataset = load_quadcopter_dataset(data_file)

    # Parse the toml config file and train
    trained_model = train_from_config_and_checkpoint(config_file, ckpt_dir, dataset)
