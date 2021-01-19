#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import gpflow as gpf
from mogpe.training import train_from_config_and_dataset


def load_quadcopter_dataset(filename, standardise=False):
    data = np.load(filename)
    # X = data['x']
    X = data["x"][:, 0:2]
    Y = data["y"]
    # Y = data['y'][:, 0:1]
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)

    X = tf.convert_to_tensor(X, dtype=gpf.default_float())
    Y = tf.convert_to_tensor(Y, dtype=gpf.default_float())

    # standardise input
    if standardise:
        mean_x, var_x = tf.nn.moments(X, axes=[0])
        mean_y, var_y = tf.nn.moments(Y, axes=[0])
        X = (X - mean_x) / tf.sqrt(var_x)
        Y = (Y - mean_y) / tf.sqrt(var_y)
    data = (X, Y)
    return data


# Set path to data set npz file
data_file = "./data/quad_sim_const_action_scenario_1.npz"

# Set path to training config
config_file = "./configs/config_2_experts.toml"
# config_file = "./configs/config_3_experts.toml"

# Load mcycle data set
dataset = load_quadcopter_dataset(data_file)
X, Y = dataset

# Parse the toml config file and train
trained_model = train_from_config_and_dataset(config_file, dataset)
gpf.utilities.print_summary(trained_model)
