import json
from gpflow.utilities import print_summary
from trainer import Trainer

from utils.config import init_model_from_config
from utils.data import load_mixture_dataset
from utils.training import save_model, custom_dir

# json_file = '../../configs/figure-3a.json'
json_file = '../../configs/figure-3b.json'
data_file = '../../data/processed/artificial-data-used-in-paper.npz'

with open(json_file, 'r') as config_file:
    config_dict = json.load(config_file)

data, F, prob_a_0 = load_mixture_dataset(filename=data_file, standardise=False)

X, Y = data
output_dim = Y.shape[1]

# Initialise the exerpts, gating network and model using the config file
model = init_model_from_config(X, output_dim, config_dict)
print_summary(model)

# initialise the trainer
custom_dir, log_dir_date = custom_dir('artificial', config_dict)
logging_dir = "../../models/logs/" + custom_dir
trainer = Trainer(model,
                  data,
                  batch_size=config_dict['batch_size'],
                  log_dir=logging_dir,
                  log_dir_date=log_dir_date)

# trainer.simple_training_loop(epochs=config_dict['num_epochs'],
#                              logging_epoch_freq=10)

trainer.monitor_training_loop(epochs=config_dict['num_epochs'],
                              fast_period=config_dict['fast_period'],
                              slow_period=config_dict['slow_period'],
                              logging_epoch_freq=100)

save_model_dir = "../../models/saved_model/" + custom_dir
save_model(model, save_model_dir)
print('Finished training and saved model')
print_summary(model)

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

# x_min = X.numpy().min() * 1.2
# x_max = X.numpy().max() * 1.2
# test_input = np.linspace(x_min, x_max, 100).reshape(-1, 1)
# plotter = Plotter(model, X.numpy(), Y.numpy(), test_input)
# plotter.plot_y_moment_matched()
# plotter.plot_ys()
# plt.show()
