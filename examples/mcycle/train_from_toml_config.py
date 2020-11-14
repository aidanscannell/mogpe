#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd

from gpflow import default_float

from mogpe.training import train_from_config_and_dataset


def load_mcycle_dataset(filename='./mcycle.csv'):
    df = pd.read_csv(filename, sep=',')
    X = pd.to_numeric(df['times']).to_numpy().reshape(-1, 1)
    Y = pd.to_numeric(df['accel']).to_numpy().reshape(-1, 1)

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
    return data


# Set path to data set csv file
data_file = './data/mcycle.csv'

# Set path to training config
config_file = './configs/config_2_experts.toml'
# config_file = './configs/config_3_experts.toml'

# Load mcycle data set
dataset = load_mcycle_dataset(data_file)

# Parse the toml config file and train
trained_model = train_from_config_and_dataset(config_file, dataset)
