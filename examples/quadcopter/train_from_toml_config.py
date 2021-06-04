#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from gpflow import default_float
from mogpe.training import train_from_config_and_dataset

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


def load_quadcopter_dataset(filename, standardise=False):
    data = np.load(filename)
    X = data["x"]
    Y = data["y"][:, 0:2]
    # Y = data["y"][:, 0:1]
    # Y = data['y'][:, 0:3]
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)

    # remove some data points
    def trim_dataset(X, Y, x1_low, x2_low, x1_high, x2_high):
        mask_0 = X[:, 0] < x1_low
        mask_1 = X[:, 1] < x2_low
        mask_2 = X[:, 0] > x1_high
        mask_3 = X[:, 1] > x2_high
        mask = mask_0 | mask_1 | mask_2 | mask_3
        X_partial = X[mask, :]
        Y_partial = Y[mask, :]
        return X_partial, Y_partial

    # X, Y = trim_dataset(X, Y, x1_low=-1., x2_low=-1., x1_high=1., x2_high=3.)
    X, Y = trim_dataset(
        X, Y, x1_low=x1_low, x2_low=x2_low, x1_high=x1_high, x2_high=x2_high
    )

    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    print("Trimmed input data shape: ", X.shape)
    print("Trimmed output data shape: ", Y.shape)

    import matplotlib.pyplot as plt

    plt.quiver(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1])
    plt.show()

    # standardise input
    mean_x, var_x = tf.nn.moments(X, axes=[0])
    mean_y, var_y = tf.nn.moments(Y, axes=[0])
    X = (X - mean_x) / tf.sqrt(var_x)
    Y = (Y - mean_y) / tf.sqrt(var_y)
    data = (X, Y)
    return data


# Set path to data set npz file
# # data_file = './data/quadcopter_data.npz'
# data_file = "./data/quadcopter_data.npz"
data_file = "./data/quadcopter_data_step_20.npz"
# data_file = "./data/quadcopter_data_step_40.npz"

# Set path to training config
config_file = "./configs/config_2_experts.toml"
# config_file = './configs/config_3_experts.toml'

# Load mcycle data set
dataset = load_quadcopter_dataset(data_file)

# Parse the toml config file and train
trained_model = train_from_config_and_dataset(config_file, dataset)
