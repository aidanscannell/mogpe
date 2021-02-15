#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from gpflow import default_float

from mogpe.training import train_from_config_and_dataset


def load_mixture_dataset(
    filename="./artificial-data-used-in-paper.npz", standardise=True
):
    data = np.load(filename)
    X = data["x"]
    Y = data["y"]
    F = data["f"]
    prob_a_0 = data["prob_a_0"]

    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)

    # standardise input
    mean_x, var_x = tf.nn.moments(X, axes=[0])
    mean_y, var_y = tf.nn.moments(Y, axes=[0])
    X = (X - mean_x) / tf.sqrt(var_x)
    Y = (Y - mean_y) / tf.sqrt(var_y)
    data = (X, Y)
    return data, F, prob_a_0


# data_file = "./artificial-data-used-in-paper.npz"
# config_file = "./config_2_experts.json"
data_file = "./data/artificial-data-used-in-paper.npz"
config_file = "./configs/config_2_experts.toml"
# config_file = "../quadcopter/configs/config_2_experts.toml"

dataset, _, _ = load_mixture_dataset(data_file)
trained_model = train_from_config_and_dataset(config_file, dataset)
