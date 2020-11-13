#!/usr/bin/env python3
import json
import gpflow as gpf
import tensorflow as tf

from bunch import Bunch
from datetime import datetime
from gpflow.monitor import (ModelToTensorBoard, MonitorTaskGroup, ScalarToTensorBoard)

from mogpe.training.model_parser import parse_model
from mogpe.training.utils import training_tf_loop, monitored_training_tf_loop, monitored_training_loop, save_model
from mogpe.visualization.plotter import Plotter1D
from mogpe.visualization.plotter2D import Plotter2D


def create_tf_dataset(dataset, num_data, batch_size):
    prefetch_size = tf.data.experimental.AUTOTUNE
    shuffle_buffer_size = num_data // 2
    num_batches_per_epoch = num_data // batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_dataset = (train_dataset.repeat().prefetch(prefetch_size).shuffle(
        buffer_size=shuffle_buffer_size).batch(batch_size,
                                               drop_remainder=True))
    return train_dataset, num_batches_per_epoch


def parse_fast_tasks(fast_tasks_period, training_loss, model, log_dir):
    if fast_tasks_period > 0:
        elbo_task = ScalarToTensorBoard(log_dir, training_loss, "elbo")
        model_task = ModelToTensorBoard(log_dir, model)
        return MonitorTaskGroup([model_task, elbo_task],
                                period=fast_tasks_period)
    else:
        return None


def train_with_config_and_dataset(config_file, dataset):
    with open(config_file) as json_config:
        config_dict = json.load(json_config)
    config = Bunch(config_dict)
    log_dir = config.log_dir + '/' + datetime.now().strftime("%m-%d-%H%M%S")

    X, Y = dataset
    input_dim = X.shape[1]
    num_data = X.shape[0]
    train_dataset, num_batches_per_epoch = create_tf_dataset(dataset, num_data, config.batch_size)

    model = parse_model(config, X)
    gpf.utilities.print_summary(model)

    if input_dim == 1:
        plotter = Plotter1D(model, X, Y)
        slow_tasks = plotter.tf_monitor_task_group(log_dir, config.slow_tasks_period)
    elif input_dim == 2:
        plotter = Plotter2D(model, X, Y)
        slow_tasks = plotter.tf_monitor_task_group(log_dir,
                                                   config.slow_tasks_period)
    else:
        slow_tasks = None
    training_loss = model.training_loss_closure(iter(train_dataset))
    fast_tasks = parse_fast_tasks(config.fast_tasks_period, training_loss, model, log_dir)

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
    else:
        monitored_training_loop(model,
                                training_loss,
                                epochs=config.epochs,
                                fast_tasks=fast_tasks,
                                slow_tasks=slow_tasks,
                                num_batches_per_epoch=num_batches_per_epoch,
                                logging_epoch_freq=config.logging_epoch_freq,
                                save_dir=log_dir)
    # TODO implementing saving if save_dir specified in config file
    # if config.save_dir
    # save_model_dir = log_dir + "-gpflow_model"
    # save_model(model, save_model_dir)
    # save_param_dict(model, log_dir)
    return model
