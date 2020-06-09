import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from plotter import Plotter
from ..util import load_mixture_dataset

# configure plotting
num_test = 400
artificial_dataset_filename = '../data/artificial/artificial-1d-mixture-sin-gating-sin-expert-higher-noise.npz'
# select a gating network prior
prior_name = 'gating_cosine'
prior_name = 'gating_product_cosine_rbf'

if prior_name == 'gating_cosine':
    save_dir = './saved_model_rbf_gating_kernel'
elif prior_name == 'gating_product_cosine_rbf':
    save_dir = './saved_model_composite_gating_kernel'

# load artificial dataset
(X, Y), _, _ = load_mixture_dataset(filename=artificial_dataset_filename,
                                    standardise=False)

# load the model saved at save_dir
loaded_model = tf.saved_model.load(save_dir)

# initialise test locations
x_min = X.numpy().min() * 2
x_max = X.numpy().max() * 2
input = np.linspace(x_min, x_max, num_test).reshape(-1, 1)

# initialise the plotter for our model
plotter = Plotter(loaded_model, X, Y, input)

fig, axs = plotter.init_subplot_21()
plotter.plot_y_svmogpe_and_gating_network(fig, axs)

plotter.plot_y_svmogpe()

plt.show()
