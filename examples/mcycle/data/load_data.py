#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import default_float
from sklearn.model_selection import KFold, train_test_split


def load_mcycle_dataset(
    filename: str = "./mcycle.csv",
    plot: bool = False,
    standardise: bool = True,
    test_split_size=0.0,
    trim_coords=None,
):
    data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=(1, 2))
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)

    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)

    # standardise input
    if standardise:
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()

    if trim_coords is not None:
        mask_0 = X < trim_coords[0]
        mask_1 = X > trim_coords[1]
        mask = mask_0 | mask_1
        X = X[mask].reshape(-1, 1)
        Y = Y[mask].reshape(-1, 1)
        print("Input data shape after trim: ", X.shape)
        print("Output data shape after trim: ", Y.shape)

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
        plt.scatter(X_train, Y_train, label="Train", color="k")
        if test_split_size > 0:
            plt.scatter(X_test, Y_test, label="Test", color="r")
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


# # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# # y = np.array([1, 2, 3, 4])
# kf = KFold(n_splits=2)
# kf.get_n_splits(X)
# KFold(n_splits=2, random_state=None, shuffle=False)
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
