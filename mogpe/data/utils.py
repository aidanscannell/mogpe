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


def load_quadcopter_dataset(
        filename='../../../data/processed/quadcopter_turbulence.npz',
        # filename='../../../data/processed/quadcopter_turbulence_single_direction.npz',
        # filename='../../../data/processed/quadcopter_turbulence_single_direction_with_reversed.npz',
        standardise=False):
    data = np.load(filename)
    X = data['x']
    # Y = data['y'][:, 0:1]
    # Y = data['y'][:, 0:3]
    # X = data['x'][:, 0:1]
    Y = data['y'][:, 0:2]
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)

    # remove some data points
    def trim_dataset(X, Y, x1_low=-3., x2_low=-3., x1_high=0., x2_high=-1.):
        mask_0 = X[:, 0] < x1_low
        mask_1 = X[:, 1] < x2_low
        mask_2 = X[:, 0] > x1_high
        mask_3 = X[:, 1] > x2_high
        mask = mask_0 | mask_1 | mask_2 | mask_3
        X_partial = X[mask, :]
        Y_partial = Y[mask, :]
        x1 = [x1_low, x1_low, x1_high, x1_high, x1_low]
        x2 = [x2_low, x2_high, x2_high, x2_low, x2_low]
        X_missing = [x1, x2]

        print("New data shape:", Y_partial.shape)
        return X_partial, Y_partial

    # X, Y = trim_dataset(X, Y, x1_low=-3., x2_low=-3., x1_high=0., x2_high=-1.)
    # X, Y = trim_dataset(X, Y, x1_low=-1., x2_low=-0.5, x1_high=0., x2_high=1.5)
    # X, Y = trim_dataset(X, Y, x1_low=-1., x2_low=-0.5, x1_high=1., x2_high=1.5)
    # X, Y = trim_dataset(X, Y, x1_low=-0.5, x2_low=-3., x1_high=1., x2_high=1.)
    # X, Y = trim_dataset(X, Y, x1_low=-1., x2_low=-3., x1_high=2., x2_high=-0.5)
    # X, Y = trim_dataset(X, Y, x1_low=-1., x2_low=-3., x1_high=1., x2_high=0.)
    X, Y = trim_dataset(X,
                        Y,
                        x1_low=-3.,
                        x2_low=-0.5,
                        x1_high=-0.5,
                        x2_high=1.7)

    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # standardise input
    if standardise:
        X, Y = standardise_data(X, Y)
    data = (X, Y)
    return data
