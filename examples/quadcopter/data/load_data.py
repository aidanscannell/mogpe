#!/usr/bin/env python3
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import default_float
from sklearn.model_selection import KFold, train_test_split

# Define input region (rectangle) to remove data from.
# This is done to test the models ability to capture epistemic unc.
# coords = np.array([[-1.0, 1.0], [-1.0, 3.0]])
# x1_low = -1.0
# x1_high = 1.0
# x2_low = -1.0
# x2_high = 3.0
# x1_low = 0.0
# x1_high = 1.0
# x2_low = 0.0
# x2_high = 3.0


def load_quadcopter_dataset(
    filename: str,
    trim_coords: Optional[list] = None,  # [x1_low, x1_high, x2]
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    plot: Optional[bool] = False,
    standardise: Optional[bool] = True,
    test_split_size=0.0,
):
    data = np.load(filename)
    if num_inputs is not None:
        X = data["x"][:, 0:num_inputs]
    else:
        X = data["x"]
    if num_outputs is not None:
        Y = data["y"][:, 0:num_outputs]
    else:
        Y = data["y"]
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)

    if standardise:
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()

    # remove some data points
    # def trim_dataset(X, Y, x1_low, x2_low, x1_high, x2_high):
    def trim_dataset(X, Y, trim_coords):
        mask_0 = X[:, 0] < trim_coords[0][0]
        mask_1 = X[:, 1] < trim_coords[0][1]
        mask_2 = X[:, 0] > trim_coords[1][0]
        mask_3 = X[:, 1] > trim_coords[1][1]
        # mask_0 = X[:, 0] < x1_low
        # mask_1 = X[:, 1] < x2_low
        # mask_2 = X[:, 0] > x1_high
        # mask_3 = X[:, 1] > x2_high
        mask = mask_0 | mask_1 | mask_2 | mask_3
        X_partial = X[mask, :]
        Y_partial = Y[mask, :]
        return X_partial, Y_partial

    if trim_coords is not None:
        print("trim_coords")
        print(trim_coords)
        # X, Y = trim_dataset(X, Y, x1_low=-1., x2_low=-1., x1_high=1., x2_high=3.)
        X, Y = trim_dataset(X, Y, trim_coords)
        # X, Y, x1_low=x1_low, x2_low=x2_low, x1_high=x1_high, x2_high=x2_high

    # X = tf.convert_to_tensor(X, dtype=default_float())
    # Y = tf.convert_to_tensor(Y, dtype=default_float())
    print("Trimmed input data shape: ", X.shape)
    print("Trimmed output data shape: ", Y.shape)

    # standardise input
    # if standardise:
    #     mean_x, var_x = tf.nn.moments(X, axes=[0])
    #     mean_y, var_y = tf.nn.moments(Y, axes=[0])
    #     X = (X - mean_x) / tf.sqrt(var_x)
    #     Y = (Y - mean_y) / tf.sqrt(var_y)

    # if plot:
    #     plt.quiver(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1])
    #     plt.show()
    # data = (X, Y)
    # return data

    if test_split_size > 0:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_split_size, random_state=42
        )
        X_test = tf.convert_to_tensor(X_test, dtype=default_float())
        Y_test = tf.convert_to_tensor(Y_test, dtype=default_float())
        test_data = (X_test, Y_test)
    else:
        X_train, Y_train = X, Y

    X_train = tf.convert_to_tensor(X_train, dtype=default_float())
    Y_train = tf.convert_to_tensor(Y_train, dtype=default_float())
    train_data = (X_train, Y_train)

    if plot:
        plt.quiver(
            X_train[:, 0],
            X_train[:, 1],
            Y_train[:, 0],
            Y_train[:, 1],
            label="Train",
            color="k",
        )
        if test_split_size > 0:
            plt.quiver(
                X_test[:, 0],
                X_test[:, 1],
                Y_test[:, 0],
                Y_test[:, 1],
                label="Test",
                color="r",
            )
        plt.legend()
        plt.show()

    print("Train input data shape: ", X_train.shape)
    print("Train output data shape: ", Y_train.shape)
    if test_split_size > 0:
        print("Test input data shape: ", X_test.shape)
        print("Test output data shape: ", Y_test.shape)
        return train_data, test_data
    else:
        return train_data, None
