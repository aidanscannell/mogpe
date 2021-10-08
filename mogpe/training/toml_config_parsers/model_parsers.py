#!/usr/bin/env python3
import gpflow as gpf
import mogpe
import numpy as np
import tensorflow as tf
from config import config_from_dict, config_from_toml
from gpflow import default_float
from gpflow.base import Parameter
from gpflow.likelihoods import Bernoulli, Softmax
from gpflow.utilities import positive
from mogpe.experts import SVGPExpert, SVGPExperts
from mogpe.gating_networks import SVGPGatingNetwork
from mogpe.mixture_of_experts import MixtureOfSVGPExperts

DEFAULT_VARIANCE_LOWER_BOUND = 1e-6


def parse_lengthscale(params_cfg, input_dim, active_dims=None):
    if active_dims is not None:
        input_dim = len(active_dims)
    try:
        return tf.convert_to_tensor(
            [params_cfg.lengthscale] * input_dim, dtype=default_float()
        )
    except AttributeError:
        print("No lengthscale in config so setting it to 1.0")
        return tf.convert_to_tensor([1.0] * input_dim, dtype=default_float())


def parse_variance(params_cfg):
    try:
        return tf.Variable([params_cfg.variance], dtype=default_float())
    except AttributeError:
        print("No variance in config so setting it to 1.0")
        return 1.0


def parse_rbf_kernel(kernel_cfg, input_dim):
    active_dims = parse_active_dims(kernel_cfg)
    lengthscale = parse_lengthscale(kernel_cfg.params, input_dim, active_dims)
    variance = parse_variance(kernel_cfg.params)
    return gpf.kernels.RBF(
        lengthscales=lengthscale, variance=variance, active_dims=active_dims
    )


def parse_cosine_kernel(kernel_cfg, input_dim):
    lengthscale = parse_lengthscale(kernel_cfg.params, input_dim)
    variance = parse_variance(kernel_cfg.params)
    active_dims = parse_active_dims(kernel_cfg)
    return gpf.kernels.Cosine(
        lengthscales=lengthscale, variance=variance, active_dims=active_dims
    )


def parse_prod_kernel(kernel_cfg, input_dim):
    kern = None
    for kernel in kernel_cfg.product:
        if kern is None:
            kern = parse_single_kernel(kernel_cfg, input_dim)
        else:
            kern *= parse_single_kernel(kernel_cfg, input_dim)
    return kern


def parse_single_kernel(kernel_cfg, input_dim):
    if kernel_cfg.name == "product":
        kernel = parse_prod_kernel(kernel_cfg, input_dim)
    elif kernel_cfg.name == "sum":
        # TODO
        kernel = parse_sum_kernel(kernel_cfg, input_dim)
    elif kernel_cfg.name == "rbf":
        kernel = parse_rbf_kernel(kernel_cfg, input_dim)
    elif kernel_cfg.name == "cosine":
        kernel = parse_cosine_kernel(kernel_cfg, input_dim)
    else:
        raise NotImplementedError(
            "Kernel name " + kernel.name + " cannot be instantiated using toml config"
        )
    return kernel


def parse_multioutput_kernel(kernels_cfg, input_dim, output_dim):
    kern_list = []
    for kernel_cfg in kernels_cfg:
        kern_list.append(parse_single_kernel(config_from_dict(kernel_cfg), input_dim))
    return gpf.kernels.SeparateIndependent(kern_list)


def parse_kernel(kernels_cfg, input_dim, output_dim):
    if isinstance(kernels_cfg, list):
        # kernel params SPECIFIED for each output dimension
        return parse_multioutput_kernel(kernels_cfg, input_dim, output_dim)
    else:
        if output_dim > 1:
            # kernel params NOT SPECIFIED for each output dimension (use same params)
            kern_list = []
            for _ in range(output_dim):
                kern_list.append(
                    parse_single_kernel(config_from_dict(kernels_cfg), input_dim)
                )
            return gpf.kernels.SeparateIndependent(kern_list)
        else:
            return parse_single_kernel(config_from_dict(kernels_cfg), input_dim)


def parse_gaussian_likelihood(likelihood_cfg, output_dim):
    if output_dim > 1:
        try:
            params = likelihood_cfg.params
            variance = parse_variance(params)
            # variance = tf.Variable(variance, dtype=default_float())
            variance = Parameter(
                variance,
                transform=positive(lower=DEFAULT_VARIANCE_LOWER_BOUND),
                dtype=default_float(),
            )
            return mogpe.gps.likelihoods.Gaussian(variance=variance)
        except AttributeError:
            print("No variance found for Gaussian likelihood")
            return mogpe.gps.likelihoods.Gaussian()
    else:
        try:
            params = likelihood_cfg.params
            variance = parse_variance(params)
            return gpf.likelihoods.Gaussian(variance=variance)
        except AttributeError:
            print("No variance found for Gaussian likelihood")
            return gpf.likelihoods.Gaussian()


def parse_likelihood(likelihood_cfg, output_dim):
    if likelihood_cfg.name == "gaussian":
        return parse_gaussian_likelihood(likelihood_cfg, output_dim)
    else:
        raise NotImplementedError(
            "This likelihood cannot be instantiated using toml config"
        )


def parse_q_diag(inducing_points_cfg):
    try:
        return inducing_points_cfg.q_diag
    except AttributeError:
        print("q_diag not found in toml config, so setting as False")
        return False


def parse_q_mu(inducing_points_cfg, output_dim):
    try:
        q_mu_mean = inducing_points_cfg.q_mu.mean
    except AttributeError:
        print("q_mu.mean not found in toml config")
        return None
    try:
        q_mu_var = inducing_points_cfg.q_mu.var
    except AttributeError:
        print("q_mu.var not found in toml config")
        return None
    try:
        num_inducing = inducing_points_cfg.num_inducing
    except AttributeError:
        print("num_inducing not found in toml config")
        return None
    return (
        q_mu_mean * np.ones((num_inducing, 1))
        + np.random.randn(num_inducing, output_dim) * q_mu_var
    )


def parse_q_sqrt(inducing_points_cfg, output_dim):
    try:
        q_sqrt = inducing_points_cfg.q_sqrt
    except AttributeError:
        print("q_sqrt not found in toml config")
        return None
    try:
        num_inducing = inducing_points_cfg.num_inducing
    except AttributeError:
        print("num_inducing not found in toml config")
        return None
    return np.array(
        [
            q_sqrt * np.eye(num_inducing, dtype=default_float())
            for _ in range(output_dim)
        ]
    )


def parse_inducing_points(expert_cfg, output_dim):
    try:
        inducing_points_cfg = expert_cfg.inducing_points
    except AttributeError:
        print(
            "inducing_points not found in toml config, setting q_mu=None, q_sqrt=None, q_diag=False"
        )
        return None, None, False
    q_mu = parse_q_mu(inducing_points_cfg, output_dim)
    q_sqrt = parse_q_sqrt(inducing_points_cfg, output_dim)
    q_diag = parse_q_diag(inducing_points_cfg)
    return q_mu, q_sqrt, q_diag


def parse_single_inducing_variable(svgp_cfg, X):
    input_dim = X.shape[1]
    try:
        num_inducing = svgp_cfg.inducing_points.num_inducing
    except AttributeError:
        print("num_inducing not found in toml config so setting as num_data/4")
        num_inducing = int(X.shape[0] / 4)

    # TODO use subest of X to initiate inducing inputs
    if not isinstance(X, np.ndarray):
        X = X.numpy()
    idx = np.random.choice(range(X.shape[0]), size=num_inducing, replace=False)

    inducing_inputs = X[idx, :].reshape(num_inducing, input_dim)
    return gpf.inducing_variables.InducingPoints(inducing_inputs)


def parse_inducing_variable(svgp_cfg, X, output_dim, num_sets=1):
    if output_dim == 1:
        return parse_single_inducing_variable(svgp_cfg, X)
    elif output_dim > 1:
        if output_dim == num_sets:
            inducing_inputs_list = []
            for _ in range(num_sets):
                inducing_inputs_list.append(parse_single_inducing_variable(svgp_cfg, X))
            return gpf.inducing_variables.SeparateIndependentInducingVariables(
                inducing_inputs_list
            )
        else:
            return gpf.inducing_variables.SharedIndependentInducingVariables(
                parse_single_inducing_variable(svgp_cfg, X)
            )
    else:
        raise NotImplementedError("output_dim should be more than 1")


def parse_constant_mean_function(mean_function_cfg):
    try:
        constant = mean_function_cfg.params.constant
    except AttributeError:
        print("No mean_function.constant in toml config so using default for gpflow")
        constant = None
    return gpf.mean_functions.Constant(constant)


def parse_mean_function(svgp_cfg):
    try:
        name = svgp_cfg.mean_function.name
    except AttributeError:
        print(
            "No mean_function.name in toml config so using gpflow.mean_functions.Zero()"
        )
        return gpf.mean_functions.Zero()
    if name == "constant":
        return parse_constant_mean_function(svgp_cfg.mean_function)
    elif name == "zero":
        return gpf.mean_functions.Zero()
    else:
        raise NotImplementedError(
            "This mean function cannot be instantiated using toml config (only zero and constant)"
        )


def parse_whiten(svgp_cfg):
    try:
        return svgp_cfg.whiten
    except AttributeError:
        print("No whiten in toml config so setting whiten=True")
        return True


def parse_active_dims(kernel_cfg):
    try:
        return kernel_cfg.active_dims
    except AttributeError:
        print("No active_dims in toml config so setting active_dims=None")
        return None


def parse_num_samples(cfg):
    try:
        return cfg.num_samples
    except AttributeError:
        print("No num_samples specified in toml config so using num_samples=1")
        return 1


def parse_num_experts(cfg):
    try:
        return cfg.num_experts
    except AttributeError:
        raise NotImplementedError("num_experts not specified in toml config")


def parse_bound(cfg):
    try:
        if cfg.bound == "further" or cfg.bound == "tight":
            return cfg.bound
        else:
            print(
                "Bound in toml config should be either 'further' or 'tight',  setting bound='further'"
            )
            return "further"
    except AttributeError:
        print("No bound in toml config so setting bound='further'")
        return "further"


def parse_gating_network(cfg, X):
    active_dims = parse_active_dims(cfg.gating_network)
    if active_dims is not None:
        X_active = []
        for dim in active_dims:
            X_active.append(X[:, dim])
        X = np.stack(X_active, -1)

    num_data, input_dim = X.shape
    num_experts = parse_num_experts(cfg)

    if num_experts > 2:
        likelihood = Softmax(num_experts)
        num_gating_functions = num_experts
        inducing_variable = parse_inducing_variable(
            cfg.gating_network, X, output_dim=num_experts, num_sets=1
        )
    else:
        likelihood = Bernoulli()
        num_gating_functions = 1
        inducing_variable = parse_single_inducing_variable(cfg.gating_network, X)
    kernel = parse_kernel(
        cfg.gating_network.kernel, input_dim, output_dim=num_gating_functions
    )
    q_mu, q_sqrt, q_diag = parse_inducing_points(
        cfg.gating_network, num_gating_functions
    )
    mean_function = parse_mean_function(cfg.gating_network)
    whiten = parse_whiten(cfg.gating_network)

    return SVGPGatingNetwork(
        kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        mean_function=mean_function,
        num_gating_functions=num_gating_functions,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        num_data=num_data,
    )


def parse_expert(expert_cfg, input_dim, output_dim, X):
    expert_cfg = config_from_dict(expert_cfg)
    num_data = X.shape[0]
    mean_function = parse_mean_function(expert_cfg)
    likelihood = parse_likelihood(expert_cfg.likelihood, output_dim)
    kernel = parse_kernel(expert_cfg.kernel, input_dim=input_dim, output_dim=output_dim)

    q_mu, q_sqrt, q_diag = parse_inducing_points(expert_cfg, output_dim)
    whiten = parse_whiten(expert_cfg)

    inducing_variable = parse_inducing_variable(expert_cfg, X, output_dim, num_sets=1)

    return SVGPExpert(
        kernel,
        likelihood,
        inducing_variable,
        mean_function=mean_function,
        num_latent_gps=output_dim,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        num_data=num_data,
    )


def parse_experts(experts_cfg, input_dim, output_dim, X):
    experts_list = []
    for expert_cfg in experts_cfg:
        experts_list.append(parse_expert(expert_cfg, input_dim, output_dim, X))
    return SVGPExperts(experts_list)


def MixtureOfSVGPExperts_from_toml(config_file, dataset):
    cfg = config_from_toml(config_file, read_from_file=True)

    X, Y = dataset
    num_data, input_dim = X.shape
    _, output_dim = Y.shape

    experts = parse_experts(cfg.experts, input_dim, output_dim, X)
    gating_network = parse_gating_network(cfg, X)
    bound = parse_bound(cfg)

    return MixtureOfSVGPExperts(
        gating_network=gating_network,
        experts=experts,
        num_samples=parse_num_samples(cfg),
        num_data=num_data,
        bound=bound,
    )


# class ConfigParser:
#     def __init__(self, config_file, dataset=None):
#         self.cfg = config_from_toml(config_file, read_from_file=True)
#         if dataset is not None:
#             self.X, self.Y = dataset
#             self.num_data = self.X.shape[0]
#             self.input_dim = self.X.shape[1]
#             self.output_dim = self.Y.shape[1]

#         experts = self.parse_experts(cfg.experts, num_data, X)
#         gating_network = self.parse_gating_network(cfg.gating_network, X)

#         model = MixtureOfSVGPExperts(
#             gating_network=gating_network,
#             experts=experts,
#             num_samples=self.cfg.num_samples,
#             num_data=self.cfg.num_data,
#         )
