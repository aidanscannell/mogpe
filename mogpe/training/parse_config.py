import json
import gpflow as gpf
import tensorflow as tf

from bunch import Bunch
from datetime import datetime
from gpflow.monitor import (ModelToTensorBoard, MonitorTaskGroup,
                            ScalarToTensorBoard)

from mogpe.data.utils import load_mixture_dataset, load_mcycle_dataset, load_quadcopter_dataset
from mogpe.models.utils.model_parser import parse_model
from mogpe.training.utils import training_tf_loop, monitored_training_tf_loop, monitored_training_loop
from mogpe.visualization.plotter import Plotter1D
from mogpe.visualization.plotter2D import Plotter2D


def parse_fast_tasks(fast_tasks_period, training_loss, model, log_dir):
    if fast_tasks_period > 0:
        elbo_task = ScalarToTensorBoard(log_dir, training_loss, "elbo")
        model_task = ModelToTensorBoard(log_dir, model)
        return MonitorTaskGroup([model_task, elbo_task],
                                period=fast_tasks_period)
    else:
        return None


# def parse_slow_tasks(slow_tasks_period, plotter, num_experts, log_dir):
#     if slow_tasks_period > 0:
#         return init_slow_tasks(plotter,
#                                num_experts,
#                                log_dir,
#                                slow_period=slow_tasks_period)
#     else:
#         return None


def parse_dataset(dataset_name):
    if dataset_name == 'mcycle':
        data_file = '../../data/external/mcycle.csv'
        dataset = load_mcycle_dataset(filename=data_file)
    elif dataset_name == 'artificial':
        data_file = '../../data/processed/artificial-data-used-in-paper.npz'
        dataset, _, _ = load_mixture_dataset(filename=data_file)

    elif dataset_name == 'quadcopter':
        data_file = '../../data/processed/quadcopter_turbulence.npz'
        dataset = load_quadcopter_dataset(filename=data_file)
    else:
        raise NotImplementedError('No dataset by this name.')
    return dataset


def create_tf_dataset(dataset, num_data, batch_size):
    prefetch_size = tf.data.experimental.AUTOTUNE
    shuffle_buffer_size = num_data // 2
    num_batches_per_epoch = num_data // batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_dataset = (train_dataset.repeat().prefetch(prefetch_size).shuffle(
        buffer_size=shuffle_buffer_size).batch(batch_size,
                                               drop_remainder=True))
    return train_dataset, num_batches_per_epoch


def run_config_file(config_file):
    with open(config_file) as json_config:
        config_dict = json.load(json_config)
    config = Bunch(config_dict)
    num_inducing_samples = config.num_inducing_samples

    dataset = parse_dataset(config.dataset_name)
    X, Y = dataset
    num_data = X.shape[0]
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    train_dataset, num_batches_per_epoch = create_tf_dataset(
        dataset, num_data, config.batch_size)

    model = parse_model(config, X)
    # gpf.set_trainable(model.experts.experts_list[0].likelihood.variance, False)
    # gpf.set_trainable(model.experts.experts_list[0].inducing_variable, False)
    # gpf.set_trainable(model.experts.experts_list[1].inducing_variable, False)
    # gpf.set_trainable(model.gating_network.inducing_variable, False)
    gpf.utilities.print_summary(model)

    log_dir = '../../models/logs/' + config.dataset_name + '/' + datetime.now(
    ).strftime("%m-%d-%H%M%S")

    if input_dim == 1:
        plotter = Plotter1D(model, X, Y)
        slow_tasks = plotter.tf_monitor_task_group(log_dir,
                                                   config.slow_tasks_period)
        # slow_tasks = parse_slow_tasks(config.slow_tasks_period, plotter,
        #                               config.num_experts, log_dir)
    elif input_dim == 2:
        plotter = Plotter2D(model, X, Y)
        slow_tasks = plotter.tf_monitor_task_group(log_dir,
                                                   config.slow_tasks_period)
        # slow_tasks = parse_slow_tasks(config.slow_tasks_period, plotter,
        #                               config.num_experts, log_dir)
    else:
        slow_tasks = None
    training_loss = model.training_loss_closure(iter(train_dataset))
    fast_tasks = parse_fast_tasks(config.fast_tasks_period, training_loss,
                                  model, log_dir)

    if fast_tasks is None and slow_tasks is None:
        training_tf_loop(model,
                         training_loss,
                         epochs=config.epochs,
                         num_batches_per_epoch=num_batches_per_epoch,
                         logging_epoch_freq=config.logging_epoch_freq)
    elif slow_tasks is None:
        monitored_training_tf_loop(
            model,
            training_loss,
            epochs=config.epochs,
            fast_tasks=fast_tasks,
            num_batches_per_epoch=num_batches_per_epoch,
            logging_epoch_freq=config.logging_epoch_freq)
    elif fast_tasks is None:
        raise NotImplementedError
        # monitored_training_loop(model,
        #                         training_loss,
        #                         epochs=config.epochs,
        #                         fast_tasks=fast_tasks,
        #                         slow_tasks=slow_tasks,
        #                         num_batches_per_epoch=num_batches_per_epoch,
        #                         logging_epoch_freq=config.logging_epoch_freq)
    else:
        monitored_training_loop(model,
                                training_loss,
                                epochs=config.epochs,
                                fast_tasks=fast_tasks,
                                slow_tasks=slow_tasks,
                                num_batches_per_epoch=num_batches_per_epoch,
                                logging_epoch_freq=config.logging_epoch_freq)


def parse_model_from_config_file(config_file):
    with open(config_file) as json_config:
        config_dict = json.load(json_config)
    config = Bunch(config_dict)
    dataset = parse_dataset(config.dataset_name)
    X, Y = dataset
    return parse_model(config, X)


if __name__ == "__main__":
    config_file = '../../configs/mcycle.json'
    config_file = '../../configs/artificial_2b.json'
    config_file = '../../configs/quadcopter.json'
    config_file = '../../configs/mcycle-3-experts.json'

    # model = parse_config_json(config_file)
    run_config_file(config_file)
    # gpf.utilities.print_summary(model)
    # TODO make gating_netowrk accept different likelihoods
    # TODO make size of input_dim
    # TODO make subset of X
    # TODO separate variance per output?
    # TODO separate mean function per output?
