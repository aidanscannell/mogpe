#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow import default_float


def load_mcycle_dataset(
    filename: str = "./mcycle.csv", plot: bool = False, standardise: bool = True
):
    data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=(1, 2))
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)

    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)

    # standardise input
    if standardise:
        mean_x, var_x = tf.nn.moments(X, axes=[0])
        mean_y, var_y = tf.nn.moments(Y, axes=[0])
        X = (X - mean_x) / tf.sqrt(var_x)
        Y = (Y - mean_y) / tf.sqrt(var_y)
    if plot:
        plt.scatter(X, Y)
        plt.show()
    data = (X, Y)
    return data
