import gpflow as gpf
import numpy as np
import tensorflow as tf
from experts import ExpertsSeparate
from gating_network import GatingNetwork
from gpflow import default_float
# from svmogpe import SVMoGPE
from src.models.utils.model import init_inducing_variables


def init_expert_from_config(X, output_dim, config_dict):
    num_data = X.shape[0]
    input_dim = X.shape[1]
    if config_dict['num_inducing'] == "None":
        num_inducing = int(np.ceil(np.log(num_data)**input_dim))
    else:
        num_inducing = config_dict['num_inducing']

    inducing_variable = init_inducing_variables(X, num_inducing)

    if config_dict['mean_func'] == 'constant':
        mean_func = gpf.mean_functions.Constant()
    else:
        mean_func = gpf.mean_functions.Zero()

    noise_var = tf.convert_to_tensor([config_dict['noise_var']] * output_dim,
                                     dtype=default_float())

    if config_dict['kern'] == 'cosine':
        kernel = gpf.kernels.Cosine()
        # kernel.lengthscales.assign(config_dict['lengthscale'])
        # kernel.variance.assign(config_dict['kern_var'])
        kernel.lengthscales.assign(3)
        kernel.variance.assign(3)
    elif config_dict['kern'] == 'rbf':
        # lengthscale = tf.convert_to_tensor([config_dict['lengthscale']] *
        #                                    input_dim,
        #                                    dtype=default_float())
        lengthscale = tf.convert_to_tensor([10.] * input_dim,
                                           dtype=default_float())
        kernel = gpf.kernels.RBF(lengthscales=lengthscale)

    # TODO - correct this
    kern_list = [kernel for _ in range(output_dim)]
    # Create multioutput kernel from kernel list
    kernel = gpf.kernels.SeparateIndependent(kern_list)

    return inducing_variable, mean_func, kernel, noise_var


def init_experts_from_config(X, output_dim, config_dict):
    inducing_variable_1, mean_func_1, kernel_1, noise_var_1 = init_expert_from_config(
        X, output_dim, config_dict['expert_1'])
    inducing_variable_2, mean_func_2, kernel_2, noise_var_2 = init_expert_from_config(
        X, output_dim, config_dict['expert_2'])
    inducing_variables = [inducing_variable_1, inducing_variable_2]
    kernels = [kernel_1, kernel_2]
    noise_vars = [noise_var_1, noise_var_2]
    mean_funcs = [mean_func_1, mean_func_2]

    if config_dict['num_samples_expert_expectation'] == "None":
        num_samples = None
    else:
        num_samples = int(config_dict['num_samples_expert_expectation'])

    experts = ExpertsSeparate(inducing_variables,
                              output_dim,
                              kernels,
                              mean_funcs,
                              num_samples=num_samples,
                              noise_vars=noise_vars)
    return experts


def init_gating_from_config(X, output_dim, config_dict):
    num_data = X.shape[0]
    input_dim = X.shape[1]
    if config_dict['num_inducing'] == "None":
        num_inducing = int(np.ceil(np.log(num_data)**input_dim))
    else:
        num_inducing = config_dict['num_inducing']

    # TODO - make these configurable in config/example.json
    # q_mu = np.zeros((num_inducing, 1)) + np.random.randn(num_inducing, 1) * 4
    # q_sqrt = np.array(
    #     [20 * np.eye(num_inducing, dtype=default_float()) for _ in range(1)])
    q_mu = np.zeros((num_inducing, 1)) + np.random.randn(num_inducing, 1) * 2
    q_sqrt = np.array(
        [10 * np.eye(num_inducing, dtype=default_float()) for _ in range(1)])

    inducing_variable = init_inducing_variables(X, num_inducing)

    if config_dict['kern'] == 'cosine':
        kernel = gpf.kernels.Cosine()
        kernel.lengthscales.assign(config_dict['lengthscale'])
        kernel.variance.assign(config_dict['kern_var'])
    elif config_dict['kern'] == 'prod-rbf-cosine':
        lengthscale_cos = tf.convert_to_tensor([config_dict['lengthscale']] *
                                               input_dim,
                                               dtype=default_float())
        kernel_cos = gpf.kernels.Cosine(lengthscales=lengthscale_cos)
        lengthscale_rbf = tf.convert_to_tensor([config_dict['lengthscale']] *
                                               input_dim,
                                               dtype=default_float())
        kernel_rbf = gpf.kernels.SquaredExponential(
            lengthscales=lengthscale_rbf)
        kernel = kernel_rbf * kernel_cos
    elif config_dict['kern'] == 'rbf':
        lengthscale = tf.convert_to_tensor([config_dict['lengthscale']] *
                                           input_dim,
                                           dtype=default_float())
        kernel = gpf.kernels.RBF(lengthscales=lengthscale)

    if config_dict['mean_func'] == 'constant':
        mean_func = gpf.mean_functions.Constant()
    else:
        mean_func = gpf.mean_functions.Zero()

    return GatingNetwork(kernel,
                         inducing_variable,
                         mean_func,
                         num_latent_gps=1,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt,
                         num_data=num_data)


# def init_model_from_config(X, output_dim, config_dict):
#     num_data = X.shape[0]
#     input_dim = X.shape[1]
#     experts = init_experts_from_config(X, output_dim, config_dict)
#     gating_network = init_gating_from_config(X, output_dim,
#                                              config_dict['gating'])
#     return SVMoGPE(input_dim,
#                    output_dim,
#                    experts=experts,
#                    gating_network=gating_network)
