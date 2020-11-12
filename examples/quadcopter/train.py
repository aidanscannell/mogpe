#!/usr/bin/env python3


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

    X, Y = trim_dataset(X, Y, x1_low=-1., x2_low=-1., x1_high=1., x2_high=3.)

    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    # import matplotlib.pyplot as plt
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    # standardise input
    if standardise:
        X, Y = standardise_data(X, Y)
    data = (X, Y)
    return data
