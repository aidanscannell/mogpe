#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd

from gpflow import default_float

from mogpe.training import train_from_config_and_dataset, create_mosvgpe_model_from_config
from mogpe.helpers import Plotter1D


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
# trained_model = train_from_config_and_dataset(config_file, dataset)

X, Y = dataset
# plotter = Plotter1D(trained_model, X, Y)

model = create_mosvgpe_model_from_config(config_file, X=X)
import numpy as np
import matplotlib.pyplot as plt
num_test = 100
test_inputs = np.linspace(-2.5, 3., num_test).reshape(num_test, 1)
fig, ax = plt.subplots(1, 1)
f_means, f_vars = model.predict_experts_fs(test_inputs, full_cov=False)
print('fmeans')
print(f_means.shape)
print(f_vars.shape)



params = {
    # 'axes.labelsize': 30,
    # 'font.size': 30,
    # 'legend.fontsize': 20,
    # 'xtick.labelsize': 30,
    # 'ytick.labelsize': 30,
    'text.usetex': True,
}
plt.rcParams.update(params)
alpha = 0.4
ax.scatter(X, Y, marker='x', color='k', alpha=alpha)
def plot_gp(fig, ax, mean, var, color="b", label=""):
    ax.plot(test_inputs, mean, color=color, lw=2, label=label)
    ax.fill_between(
        test_inputs[:, 0],
        mean - 1.96 * np.sqrt(var),
        mean + 1.96 * np.sqrt(var),
        color=color,
        alpha=0.2,
    )

colors = ['magenta', 'cyan']
for k, expert in enumerate(model.experts.experts_list):
    label = "Expert " + str(k)
    plot_gp(fig, ax, f_means[:,0, k], f_vars[:,0, k], color=colors[k], label=label)

plt.show()
