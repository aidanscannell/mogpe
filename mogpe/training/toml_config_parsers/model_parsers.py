#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import tensorflow as tf
import toml
from bunch import Bunch
from gpflow import default_float
from mogpe.experts import SVGPExpert, SVGPExperts
from mogpe.gating_networks import (
    SVGPGatingFunction,
    SVGPGatingNetworkBinary,
    SVGPGatingNetworkMulti,
)
from mogpe.mixture_of_experts import MixtureOfSVGPExperts


def parse_kernel(kernel, input_dim, output_dim):
    kern_list = []
    for _ in range(output_dim):
        kern_list.append(parse_single_kernel(kernel, input_dim))
    # TODO - correct this
    # Create multioutput kernel from kernel list
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    return kernel


def parse_single_kernel(kernel, input_dim):
    if kernel.name == "product":
        kernel = parse_prod_kernel(kernel, input_dim)
    elif kernel.name == "sum":
        # TODO
        kernel = parse_sum_kernel(kernel, input_dim)
    elif kernel.name == "rbf":
        kernel = parse_rbf_kernel(kernel, input_dim)
    elif kernel.name == "cosine":
        kernel = parse_cosine_kernel(kernel, input_dim)
    else:
        raise NotImplementedError(
            "Kernel name " + kernel.name + " cannot be instantiated using toml config"
        )
    return kernel


def parse_prod_kernel(kernel, input_dim):
    kern = None
    try:
        for kernel in kernel.product:
            kernel = Bunch(kernel)
            if kern is None:
                kern = parse_single_kernel(kernel, input_dim)
            else:
                kern *= parse_single_kernel(kernel, input_dim)
        return kern
    except:
        raise NotImplementedError("Error parsing product of kernels")


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
        return tf.convert_to_tensor(
            [params.lengthscale] * input_dim, dtype=default_float()
        )
    except:
        return tf.convert_to_tensor([1.0] * input_dim, dtype=default_float())


def parse_variance(params):
    try:
        return params.variance
    except:
        return 1.0


def parse_likelihood(likelihood):
    if likelihood.name == "gaussian":
        return parse_gaussian_likelihood(likelihood)
    else:
        raise NotImplementedError(
            "This likelihood cannot be instantiated using json config"
        )
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
        q_mu, q_sqrt = parse_inducing_output(Bunch(inducing_points), output_dim)
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
        return (
            q_mu.mean * np.ones((inducing_points.num_inducing, 1))
            + np.random.randn(inducing_points.num_inducing, output_dim) * q_mu.var
        )
    except:
        return None


def parse_q_sqrt(inducing_points, output_dim):
    try:
        return np.array(
            [
                inducing_points.q_sqrt
                * np.eye(inducing_points.num_inducing, dtype=default_float())
                for _ in range(output_dim)
            ]
        )
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
        if not isinstance(X, np.ndarray):
            X = X.numpy()

        idx = np.random.choice(
            range(X.shape[0]), size=inducing_points.num_inducing, replace=False
        )
        inducing_inputs = X[idx, :].reshape(inducing_points.num_inducing, input_dim)
        return gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(inducing_inputs)
        )
    except:
        inducing_points = Bunch(expert.inducing_points)
        X = []
        for _ in range(input_dim):
            X.append(np.linspace(0, 1, inducing_points.num_inducing))
        return gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(np.array(X).T)
        )


def parse_mean_function(expert):
    try:
        if expert.mean_function == "constant":
            return gpf.mean_functions.Constant()
        elif expert.mean_function == "zero":
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


def parse_num_inducing_samples(config):
    try:
        return config.num_inducing_samples
    except:
        print(
            "num_inducing_samples not specified in toml config so using num_inducing_samples=1"
        )
        return 1


def parse_num_experts(config):
    try:
        return config.num_experts
    except:
        raise NotImplementedError("num_expets not specified in toml config")


def parse_gating_function(gating_function, input_dim, output_dim, num_data, X):
    # TODO remove this output dim hack and fix code
    output_dim = 1
    mean_function = parse_mean_function(gating_function)
    kernel = parse_kernel(
        Bunch(gating_function.kernel), input_dim=input_dim, output_dim=output_dim
    )

    q_mu, q_sqrt, q_diag = parse_inducing_points(gating_function, output_dim)
    whiten = parse_whiten(gating_function)
    inducing_variable = parse_inducing_variable(gating_function, input_dim, X)

    return SVGPGatingFunction(
        kernel,
        inducing_variable=inducing_variable,
        mean_function=mean_function,
        num_latent_gps=output_dim,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        num_data=num_data,
    )


def parse_binary_gating_network(gating_network, input_dim, output_dim, num_data, X):
    gating_function = parse_gating_function(
        gating_network, input_dim, output_dim, num_data, X
    )

    return SVGPGatingNetworkBinary(gating_function)


def parse_multi_gating_network(config, input_dim, output_dim, num_data, X):
    gating_function_list = []
    for gating_function in config.gating_functions:
        gating_function_list.append(
            parse_gating_function(
                Bunch(gating_function), input_dim, output_dim, num_data, X
            )
        )
    return SVGPGatingNetworkMulti(gating_function_list)


def parse_gating_network(config, X):
    num_data = parse_num_data(config)
    num_experts = parse_num_experts(config)
    if num_experts > 2:
        return parse_multi_gating_network(
            config, config.input_dim, config.output_dim, num_data, X
        )
    else:
        try:
            gating_network = Bunch(config.gating_functions[0])
        except KeyError:
            gating_network = Bunch(config.gating_functions)

        return parse_binary_gating_network(
            gating_network, config.input_dim, config.output_dim, num_data, X
        )


def parse_expert(expert, input_dim, output_dim, num_data, X):
    mean_function = parse_mean_function(expert)
    likelihood = parse_likelihood(Bunch(expert.likelihood))
    kernel = parse_kernel(
        Bunch(expert.kernel), input_dim=input_dim, output_dim=output_dim
    )

    q_mu, q_sqrt, q_diag = parse_inducing_points(expert, output_dim)
    whiten = parse_whiten(expert)

    inducing_variable = parse_inducing_variable(expert, input_dim, X)

    # q_mu = None
    # q_sqrt = None
    # q_diag = None
    # whiten = None
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


def parse_experts(config, num_data, X):
    experts_list = []
    for expert in config.experts:
        experts_list.append(
            parse_expert(
                Bunch(expert), config.input_dim, config.output_dim, num_data, X
            )
        )
    return SVGPExperts(experts_list)


def parse_mixture_of_svgp_experts_model(config, X=None):
    # X.shape = (num_data, input_dim)
    num_data = None
    if X is None:
        try:
            num_data = config.num_data
        except:
            raise NotImplementedError(
                "Must either specify num_data in toml config or pass input data X with shape (num_data, input_dim)"
            )
    else:
        num_data = X.shape[0]
    num_inducing_samples = parse_num_inducing_samples(config)
    experts = parse_experts(config, num_data, X)
    gating_network = parse_gating_network(config, X)

    return MixtureOfSVGPExperts(
        gating_network=gating_network,
        experts=experts,
        num_inducing_samples=num_inducing_samples,
        num_data=num_data,
    )


def create_mosvgpe_model_from_config(config_file, X=None):
    with open(config_file) as toml_config:
        config_dict = toml.load(toml_config)
    config = Bunch(config_dict)
    return parse_mixture_of_svgp_experts_model(config, X)
