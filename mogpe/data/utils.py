import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow.config import default_float

# def custom_dir(dataset_name, config_dict):
#     # setup custom logging location based on config
#     if config_dict['num_samples_expert_expectation'] == "None":
#         expert_expectation = 'analytic'
#     else:
#         expert_expectation = 'sample'
#     if config_dict['add_date_to_logging_dir'] == "True":
#         log_dir_date = True
#     else:
#         log_dir_date = False
#     custom_dir = dataset_name + "/" + expert_expectation + "-f/batch_size-" + str(
#         config_dict['batch_size']) + "/num_inducing-" + str(
#             config_dict['gating']['num_inducing'])
#     return custom_dir, log_dir_date


def standardise_data(X, Y):
    mean_x, var_x = tf.nn.moments(X, axes=[0])
    mean_y, var_y = tf.nn.moments(Y, axes=[0])
    X = (X - mean_x) / tf.sqrt(var_x)
    Y = (Y - mean_y) / tf.sqrt(var_y)
    return X, Y


def load_mixture_dataset(
        filename='../../../data/processed/artificial-data-used-in-paper.npz',
        standardise=True):
    data = np.load(filename)
    X = data['x']
    Y = data['y']
    F = data['f']
    prob_a_0 = data['prob_a_0']
    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)
    if standardise:
        X, Y = standardise_data(X, Y)
    data = (X, Y)
    return data, F, prob_a_0


def load_mcycle_dataset(filename='../../../data/external/mcycle.csv'):
    df = pd.read_csv(filename, sep=',')
    X = pd.to_numeric(df['times'])
    Y = pd.to_numeric(df['accel'])
    # Y = Y + 30 * np.sin(Y)
    X = X.to_numpy().reshape(-1, 1)
    Y = Y.to_numpy().reshape(-1, 1)

    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)
    # standardise input
    X, Y = standardise_data(X, Y)
    data = (X, Y)
    return data
