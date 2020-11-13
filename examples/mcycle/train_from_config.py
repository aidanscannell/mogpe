#!/usr/bin/env python3
import gpflow as gpf
import tensorflow as tf
import pandas as pd

from gpflow import default_float

from mogpe.training.config_parser import train_with_config_and_dataset


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


data_file = './mcycle.csv'
config_file = './config_2_experts.json'

dataset = load_mcycle_dataset(data_file)
trained_model = train_with_config_and_dataset(config_file, dataset)


# save_model_dir = log_dir + "-gpflow_model"
# save_model(model, save_model_dir)

# save_param_dict(model, log_dir)
# save_model_params(model, save_dir)
# plotter.plot_model()




