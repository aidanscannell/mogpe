import gpflow as gpf
import tensorflow as tf

from datetime import datetime
from gpflow.monitor import (ModelToTensorBoard, MonitorTaskGroup,
                            ScalarToTensorBoard)

from mogpe.data.utils import load_mixture_dataset, load_mcycle_dataset
from mogpe.training.utils import training_tf_loop, monitored_training_tf_loop, monitored_training_loop, init_slow_tasks
from mogpe.models.mixture_model import init_fake_mixture
from mogpe.models.utils.parse_json_config import parse_config_json
from mogpe.visualization.plotter import Plotter1D

epochs = 20000
batch_size = 100
num_inducing_samples = 1
num_experts = 2
slow_period = 500
fast_period = 10
logging_epoch_freq = 100

dataset_name = 'artificial'
# dataset_name = 'mcycle'

if dataset_name == 'mcycle':
    config_file = '../../configs/mcycle.json'
    data_file = '../../data/external/mcycle.csv'
    dataset = load_mcycle_dataset(filename=data_file)
else:
    config_file = '../../configs/artificial_2b.json'
    # Load data set and initialise a batched tf.Dataset
    data_file = '../../data/processed/artificial-data-used-in-paper.npz'
    dataset, _, _ = load_mixture_dataset(filename=data_file)

log_dir = '../../models/logs/' + dataset_name + '/' + datetime.now().strftime(
    "%m-%d-%H%M%S")

X, Y = dataset
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
# model = init_fake_mixture(X,
#                           Y,
#                           num_experts=num_experts,
#                           num_inducing_samples=num_inducing_samples)
# gpf.utilities.print_summary(model)

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
