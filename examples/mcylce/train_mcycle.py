#!/usr/bin/env python3
import gpflow as gpf
import tensorflow as tf
import pandas as pd

from datetime import datetime
from gpflow.monitor import (ModelToTensorBoard, MonitorTaskGroup,
                            ScalarToTensorBoard)
from gpflow import default_float

from mogpe.training.utils import training_tf_loop, monitored_training_tf_loop, monitored_training_loop
from mogpe.models.utils.parse_json_config import parse_config_json
from mogpe.visualization.plotter import Plotter1D

data_file = './mcycle.csv'
config_file = './config.json'
log_dir = './logs/' + datetime.now().strftime("%m-%d-%H%M%S")

epochs = 20000
batch_size = 100
num_inducing_samples = 1
num_experts = 2
slow_period = 500
fast_period = 10
logging_epoch_freq = 100


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


# Load data set
dataset = load_mcycle_dataset(data_file)
X, Y = dataset

# Configure data set for training
num_train_data = X.shape[0]
prefetch_size = tf.data.experimental.AUTOTUNE
shuffle_buffer_size = num_train_data // 2
num_batches_per_epoch = num_train_data // batch_size
train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
train_dataset = (train_dataset.repeat().prefetch(prefetch_size).shuffle(
    buffer_size=shuffle_buffer_size).batch(batch_size))

# Initialise the mixture model
model = parse_config_json(config_file, X)
gpf.utilities.print_summary(model)

# Set up fast tasks
training_loss = model.training_loss_closure(iter(train_dataset))
elbo_task = ScalarToTensorBoard(log_dir, training_loss, "elbo")
model_task = ModelToTensorBoard(log_dir, model)
fast_tasks = MonitorTaskGroup([model_task, elbo_task], period=fast_period)

# Set the plotter instance for creating figures in slow tasks
plotter = Plotter1D(model, X, Y)
slow_tasks = init_slow_tasks(plotter,
                             num_experts,
                             log_dir,
                             slow_period=slow_period)

# training_tf_loop(model,
#                  training_loss,
#                  epochs=epochs,
#                  num_batches_per_epoch=num_batches_per_epoch,
#                  logging_epoch_freq=logging_epoch_freq)
# monitored_training_tf_loop(model,
#                            training_loss,
#                            epochs=epochs,
#                            fast_tasks=fast_tasks,
#                            num_batches_per_epoch=num_batches_per_epoch)
monitored_training_loop(model,
                        training_loss,
                        epochs=epochs,
                        fast_tasks=fast_tasks,
                        slow_tasks=slow_tasks,
                        num_batches_per_epoch=num_batches_per_epoch)

gpf.utilities.print_summary(model)
