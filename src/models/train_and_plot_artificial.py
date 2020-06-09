import json
import numpy as np
import matplotlib.pyplot as plt

from gpflow.utilities import print_summary

from trainer import Trainer
from utils.data import load_mixture_dataset
from utils.training import custom_dir, save_model
from utils.config import init_model_from_config
from src.visualization.plotter import Plotter

############################################################
# Initialise data set and model config file
############################################################
# json_file = '../../configs/figure-3a.json'
json_file = '../../configs/figure-3b.json'
data_file = '../../data/processed/artificial-data-used-in-paper.npz'

############################################################
# Load data set
############################################################
data, F, prob_a_0 = load_mixture_dataset(filename=data_file, standardise=False)
X, Y = data
output_dim = Y.shape[1]

############################################################
# Initialise model from config in json_file
############################################################
with open(json_file, 'r') as config_file:
    config_dict = json.load(config_file)

# Initialise the exerpts, gating network and model using the config file
model = init_model_from_config(X, output_dim, config_dict)
print_summary(model)

############################################################
# Training
############################################################
# initialise the trainer
custom_dir, log_dir_date = custom_dir('artificial', config_dict)
logging_dir = "../../models/logs/" + custom_dir
trainer = Trainer(model,
                  data,
                  batch_size=config_dict['batch_size'],
                  log_dir=logging_dir,
                  log_dir_date=log_dir_date)

trainer.simple_training_loop(epochs=config_dict['num_epochs'],
                             logging_epoch_freq=100)

# trainer.monitor_training_loop(epochs=config_dict['num_epochs'],
#                               fast_period=config_dict['fast_period'],
#                               slow_period=config_dict['slow_period'],
#                               logging_epoch_freq=100)

# trainer.checkpointing_training_loop(epochs=num_epochs,
#                                     logging_epoch_freq=100)
# trainer.monitor_training_loop(epochs=config_dict['num_epochs'],
#                               fast_period=fast_period,
#                               slow_period=slow_period,
#                               logging_epoch_freq=100)

# trainer.monitor_training_tf_loop(fast_period=fast_period,
#                                  epochs=num_epochs,
#                                  logging_epoch_freq=100)
# trainer.monitor_and_ckpt_training_loop(fast_period=fast_period,
#                                        epochs=num_epochs,
#                                        logging_epoch_freq=100)

############################################################
# Save the model
############################################################
save_model_dir = "../../models/saved_model/" + custom_dir
save_model(model, save_model_dir)
print('Finished training and saved model')
print_summary(model)

############################################################
# Plotting
############################################################
# initialise test locations
num_test = 400
x_min = X.numpy().min() * 2
x_max = X.numpy().max() * 2
input = np.linspace(x_min, x_max, num_test).reshape(-1, 1)

# initialise the plotter for our model
plotter = Plotter(model, X, Y, input)

fig, axs = plotter.init_subplot_21()
plotter.plot_y_svmogpe_and_gating_network(fig, axs)

plotter.plot_y_svmogpe()

plt.show()
