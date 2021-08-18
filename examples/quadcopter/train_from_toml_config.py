#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

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
    # Set path to data set npz file
    # # data_file = './data/quadcopter_data.npz'
    # data_file = "./data/quadcopter_data.npz"
    # data_file = "./data/quadcopter_data_step_20.npz"
    # data_file = "./data/quadcopter_data_step_40.npz"
    # data_file = "./quadcopter/data/quadcopter_data.npz"
    data_file = "./quadcopter/data/quadcopter_data_step_20.npz"
    # data_file = "./quadcopter/data/quadcopter_data_step_40.npz"
    data_file = "./quadcopter/data/quadcopter_data_step_20_trimmed.npz"
    data_file = "./quadcopter/data/quadcopter_data_step_40_trimmed.npz"
    # data_file = "./quadcopter/data/quadcopter_data_step_10_trimmed.npz"
    data_file = (
        "./quadcopter/data/quadcopter_data_step_40_single_direction_opposite.npz"
    )
    # data_file = "./quadcopter/data/quadcopter_data_step_10_direction_down.npz"
    data_file = "./quadcopter/data/quadcopter_data_step_20_direction_down.npz"
    # data_file = "./quadcopter/data/quadcopter_data_step_40_direction_down.npz"

    # Set path to training config
    config_file = "./quadcopter/configs/config_2_experts.toml"
    # config_file = './quadcopter/configs/config_3_experts.toml'

    # Load mcycle data set
    dataset = load_quadcopter_dataset(data_file)

    # Parse the toml config file and train
    trained_model = train_from_config_and_dataset(config_file, dataset)
