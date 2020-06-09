import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.visualization.plotter import Plotter
from src.models.utils.data import load_mixture_dataset

# configure plotting
num_test = 400
# artificial_dataset_filename = '../../data/processed/artificial-data-used-in-paper.npz'
# load_model_dir = '../../models/saved_model/artificial/prod_rbf_cosine_kernel'
artificial_dataset_filename = './data/processed/artificial-data-used-in-paper.npz'
load_model_dir = './models/saved_model/artificial/prod_rbf_cosine_kernel'

# load artificial dataset
(X, Y), _, _ = load_mixture_dataset(filename=artificial_dataset_filename,
                                    standardise=False)

# load the model saved at save_dir
loaded_model = tf.saved_model.load(load_model_dir)

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
