import json
import gpflow as gpf
import numpy as np
import tensorflow as tf

from bunch import Bunch
from datetime import datetime
from gpflow import default_float
from gpflow.monitor import (ModelToTensorBoard, MonitorTaskGroup,
                            ScalarToTensorBoard)

from mogpe.data.utils import load_mixture_dataset, load_mcycle_dataset
from mogpe.models.expert import SVGPExpert
from mogpe.models.experts import Experts
from mogpe.models.gating_network import GatingNetwork
from mogpe.models.mixture_model import GPMixtureOfExperts
from mogpe.training.utils import training_tf_loop, monitored_training_tf_loop, monitored_training_loop, init_slow_tasks
from mogpe.visualization.plotter import Plotter1D


def parse_kernel(kernel, input_dim, output_dim):
    kern_list = []
    for _ in range(output_dim):
        kern_list.append(parse_single_kernel(kernel, input_dim))
    # TODO - correct this
    # Create multioutput kernel from kernel list
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    return kernel


def parse_single_kernel(kernel, input_dim):
    if kernel.name == 'product':
        kernel = parse_prod_kernel(kernel, input_dim)
    elif kernel.name == 'sum':
        # TODO
        kernel = parse_sum_kernel(kernel, input_dim)
    elif kernel.name == 'rbf':
        kernel = parse_rbf_kernel(kernel, input_dim)
    elif kernel.name == 'cosine':
        kernel = parse_cosine_kernel(kernel, input_dim)
    else:
        raise NotImplementedError("Kernel name " + kernel.name +
                                  " cannot be instantiated using json config")
    return kernel


def parse_prod_kernel(kernel, input_dim):
    try:
        kernels = Bunch(kernel.kernels)
        print('kernels')
        print(kernels)
        for kernel in kernels:
            try:
                kern *= parse_single_kernel(kernel, input_dim)
            except:
                kern = parse_single_kernel(kernel, input_dim)
        return kern
    except:
        return gpf.kernels.RBF()


def parse_rbf_kernel(kernel, input_dim):
    try:
        params = Bunch(kernel.params)
        lengthscale = parse_lengthscale(params, input_dim)
        variance = parse_variance(params)
        return gpf.kernels.RBF(lengthscales=lengthscale, variance=variance)
    except:
        return gpf.kernels.RBF()


def parse_cosine_kernel(kernel, input_dim):
    try:
        params = Bunch(kernel.params)
        lengthscale = parse_lengthscale(params, input_dim)
        variance = parse_variance(params)
        return gpf.kernels.Cosine(lengthscales=lengthscale, variance=variance)
    except:
        return gpf.kernels.Cosine()


def parse_lengthscale(params, input_dim):
    try:
        return tf.convert_to_tensor([params.lengthscale] * input_dim,
                                    dtype=default_float())
    except:
        return tf.convert_to_tensor([1.] * input_dim, dtype=default_float())


def parse_variance(params):
    try:
        return params.variance
    except:
        return 1.0


def parse_likelihood(likelihood):
    if likelihood.name == 'gaussian':
        return parse_gaussian_likelihood(likelihood)
    else:
        raise NotImplementedError(
            "This likelihood cannot be instantiated using json config")
    return likelihood


def parse_gaussian_likelihood(likelihood):
    # TODO multioutput noise variance?
    try:
        params = Bunch(likelihood.params)
        variance = parse_variance(params)
        return gpf.likelihoods.Gaussian(variance=variance)
    except:
        return gpf.likelihoods.Gaussian()


def parse_inducing_points(expert, output_dim):
    try:
        inducing_points = Bunch(expert.inducing_points)
        q_mu, q_sqrt = parse_inducing_output(Bunch(inducing_points),
                                             output_dim)
        q_diag = parse_q_diag(Bunch(inducing_points))
        return q_mu, q_sqrt, q_diag
    except:
        return None, None, False


def parse_inducing_output(inducing_points, output_dim):
    q_mu = parse_q_mu(inducing_points, output_dim)
    q_sqrt = parse_q_sqrt(inducing_points, output_dim)
    return q_mu, q_sqrt


def parse_q_mu(inducing_points, output_dim):
    try:
        q_mu = Bunch(inducing_points.q_mu)
        return q_mu.mean * np.ones(
            (inducing_points.num_inducing, 1)) + np.random.randn(
                inducing_points.num_inducing, output_dim) * q_mu.var
    except:
        return None


def parse_q_sqrt(inducing_points, output_dim):
    try:
        return np.array([
            inducing_points.q_sqrt *
            np.eye(inducing_points.num_inducing, dtype=default_float())
            for _ in range(output_dim)
        ])
    except:
        return None


def parse_q_diag(inducing_points):
    try:
        return inducing_points.q_diag
    except:
        return False


def parse_inducing_variable(expert, input_dim, X):
    try:
        # TODO use subest of X to initiate inducing inputs
        inducing_points = Bunch(expert.inducing_points)
        X = []
        num_data = X.shape[0]
        input_dim = X.shape[1]
        idx = np.random.choice(range(num_data),
                               size=inducing_points.num_inducing,
                               replace=False)
        inducing_inputs = X[idx, ...].reshape(inducing_points.num_inducing,
                                              input_dim)
        return gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(inducing_inputs))
    except:
        inducing_points = Bunch(expert.inducing_points)
        X = []
        for _ in range(input_dim):
            X.append(np.linspace(0, 1, inducing_points.num_inducing))
        return gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(np.array(X).T))


def parse_mean_function(expert):
    try:
        if expert.mean_function == 'constant':
            return gpf.mean_functions.Constant()
        elif expert.mean_function == 'zero':
            return gpf.mean_functions.Zero()
        else:
            raise NotImplementedError(
                "This mean function cannot be instantiated using json config (only zero and constant)"
            )
    except:
        return gpf.mean_functions.Zero()


def parse_whiten(expert):
    try:
        return expert.whiten
    except:
        return True


def parse_num_data(config):
    try:
        return config.num_data
    except:
        return None


def parse_expert(expert, input_dim, output_dim, num_data, X):
    mean_function = parse_mean_function(expert)
    likelihood = parse_likelihood(Bunch(expert.likelihood))
    kernel = parse_kernel(Bunch(expert.kernel),
                          input_dim=input_dim,
                          output_dim=output_dim)

    q_mu, q_sqrt, q_diag = parse_inducing_points(expert, output_dim)
    whiten = parse_whiten(expert)

    inducing_variable = parse_inducing_variable(expert, input_dim, X)

    # q_mu = None
    # q_sqrt = None
    # q_diag = None
    # whiten = None
    return SVGPExpert(kernel,
                      likelihood,
                      inducing_variable,
                      mean_function=mean_function,
                      num_latent_gps=output_dim,
                      q_diag=q_diag,
                      q_mu=q_mu,
                      q_sqrt=q_sqrt,
                      whiten=whiten,
                      num_data=num_data)


def parse_gating_network(gating_network, input_dim, output_dim, num_data, X):
    mean_function = parse_mean_function(gating_network)
    kernel = parse_kernel(Bunch(gating_network.kernel),
                          input_dim=input_dim,
                          output_dim=output_dim)

    q_mu, q_sqrt, q_diag = parse_inducing_points(gating_network, output_dim)
    whiten = parse_whiten(gating_network)
    inducing_variable = parse_inducing_variable(gating_network, input_dim, X)

    return GatingNetwork(kernel,
                         likelihood=None,
                         inducing_variable=inducing_variable,
                         mean_function=mean_function,
                         num_latent_gps=output_dim,
                         q_diag=q_diag,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt,
                         whiten=whiten,
                         num_data=num_data)


def parse_experts(config, input_dim, output_dim, num_data, X):
    experts_list = []
    for expert in config.experts:
        experts_list.append(
            parse_expert(Bunch(expert), input_dim, output_dim, num_data, X))
    return Experts(experts_list)


def parse_fast_tasks(fast_tasks_period, log_dir, training_loss):
    if fast_tasks_period > 0:
        elbo_task = ScalarToTensorBoard(log_dir, training_loss, "elbo")
        model_task = ModelToTensorBoard(log_dir, model)
        return MonitorTaskGroup([model_task, elbo_task],
                                period=fast_tasks_period)
    else:
        return None


def parse_slow_tasks(slow_tasks_period, log_dir):
    if slow_tasks_period > 0:
        plotter = Plotter1D(model, X, Y)
        return init_slow_tasks(plotter,
                               num_experts,
                               log_dir,
                               slow_period=slow_period)
    else:
        return None


def parse_dataset(dataset_name):
    if dataset_name == 'mcycle':
        data_file = '../../data/external/mcycle.csv'
        dataset = load_mcycle_dataset(filename=data_file)
    elif dataset_name == 'artificial':
        data_file = '../../data/processed/artificial-data-used-in-paper.npz'
        dataset, _, _ = load_mixture_dataset(filename=data_file)
    else:
        raise NotImplementedError('No dataset by this name.')
    return dataset


def create_tf_dataset(dataset, num_data, batch_size):
    prefetch_size = tf.data.experimental.AUTOTUNE
    shuffle_buffer_size = num_data // 2
    num_batches_per_epoch = num_data // batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_dataset = (train_dataset.repeat().prefetch(prefetch_size).shuffle(
        buffer_size=shuffle_buffer_size).batch(batch_size))
    return train_dataset, num_batches_per_epoch


def parse_config(config):
    num_inducing_samples = config.num_inducing_samples

    dataset = parse_dataset(config.dataset_name)
    X, Y = dataset
    num_data = X.shape[0]
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    train_dataset, num_batches_per_epoch = create_tf_dataset(
        dataset, num_data, config.batch_size)

    experts = parse_experts(config, input_dim, output_dim, num_data, X)
    gating_network = Bunch(config.gating_network)
    gating_network = parse_gating_network(gating_network, input_dim,
                                          output_dim, num_data, X)
    model = GPMixtureOfExperts(gating_network=gating_network,
                               experts=experts,
                               num_inducing_samples=num_inducing_samples,
                               num_data=num_data)

    log_dir = '../../models/logs/' + config.dataset_name + '/' + datetime.now(
    ).strftime("%m-%d-%H%M%S")

    training_loss = model.training_loss_closure(iter(train_dataset))
    slow_tasks = parse_slow_tasks(config.slow_tasks_period, log_dir)
    fast_tasks = parse_fast_tasks(config.fast_tasks_period, log_dir,
                                  training_loss)

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
            slow_tasks=slow_tasks,
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


# def parse_config_json(config_file, X):
#     """Returns GPMixtureOfExperts object with config from json file

#     :param config_file: path to json config file
#     :returns: Initialised Mixture of GP Experts model
#     :rtype: GPMixtureOfExperts
#     """
#     with open(config_file) as json_config:
#         config_dict = json.load(json_config)
#     config = Bunch(config_dict)
#     return parse_config(config, X)


def run_config(config_file):
    with open(config_file) as json_config:
        config_dict = json.load(json_config)
    config = Bunch(config_dict)
    parse_config(config)


if __name__ == "__main__":
    # config_file = '../../configs/mcycle.json'
    config_file = '../../configs/artificial_2b.json'

    # model = parse_config_json(config_file)
    model = run_config(config_file)
    # gpf.utilities.print_summary(model)
    # TODO make gating_netowrk accept different likelihoods
    # TODO make size of input_dim
    # TODO make subset of X
    # TODO separate variance per output?
    # TODO separate mean function per output?
