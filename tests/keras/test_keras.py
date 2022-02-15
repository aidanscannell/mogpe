#!/usr/bin/env python3
from dataclasses import dataclass

import gpflow as gpf
import numpy as np
import pytest
import tensorflow as tf
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Constant
from mogpe.keras.experts import SVGPExpert
from mogpe.keras.gating_networks import SVGPGatingNetwork
from mogpe.keras.mixture_of_experts import MixtureOfSVGPExperts

num_data = 5
num_inducing = 4
input_dim = 2
output_dim = 1
num_experts = 2


@dataclass
class RegressionData:
    X = np.ones((num_data, input_dim))
    Y = np.ones((num_data, output_dim))


@pytest.fixture
def gaussian_likelihood():
    return Gaussian(variance=3.0, variance_lower_bound=1e-6)


@pytest.fixture
def constant_mean_function():
    return Constant(c=-2.0)
    # return [ConstantMeanFunction(c=-2.0), ConstantMeanFunction(c=output_dim * [-2.0])]


@pytest.fixture
def rbf_kernel():
    return RBF(lengthscales=input_dim * [3.0], variance=10.0)


@pytest.fixture
def inducing_variable():
    Z = np.ones((num_inducing, input_dim))
    # return SharedIndependentInducingVariablesLayer(Z)
    return InducingPoints(Z)


@pytest.fixture
def svgp_expert(
    rbf_kernel, gaussian_likelihood, constant_mean_function, inducing_variable
):
    return SVGPExpert(
        kernel=rbf_kernel,
        likelihood=gaussian_likelihood,
        mean_function=constant_mean_function,
        inducing_variable=inducing_variable,
        num_latent_gps=output_dim,
        q_diag=False,
        q_mu=None,
        q_sqrt=None,
        whiten=True,
    )


@pytest.fixture
def svgp_gating_network(rbf_kernel, constant_mean_function, inducing_variable):
    return SVGPGatingNetwork(
        kernel=rbf_kernel,
        mean_function=constant_mean_function,
        inducing_variable=inducing_variable,
        q_diag=False,
        q_mu=None,
        q_sqrt=None,
        whiten=True,
    )


@pytest.fixture
def mixture_of_svgp_experts_model(svgp_expert, svgp_gating_network):
    experts_list = [svgp_expert for _ in range(num_experts)]
    return MixtureOfSVGPExperts(
        experts_list=experts_list, gating_network=svgp_gating_network
    )


@pytest.fixture
def layers(
    mixture_of_svgp_experts_model,
    svgp_gating_network,
    svgp_expert,
    rbf_kernel,
    gaussian_likelihood,
    constant_mean_function,
    inducing_variable,
):
    return [
        mixture_of_svgp_experts_model,
        svgp_gating_network,
        svgp_expert,
        rbf_kernel,
        gaussian_likelihood,
        constant_mean_function,
        inducing_variable,
    ]


def test_serialisation(layers):
    for layer in layers:
        _test_serialisation(layer)


def _test_serialisation(layer):
    serialized = tf.keras.layers.serialize(layer)
    new = tf.keras.layers.deserialize(
        serialized, custom_objects={layer.__class__.__name__: layer.__class__}
    )
    assert isinstance(new, type(layer))


# train_dataset = (RegressionData.X, RegressionData.Y)
# train_dataset_tf = tf.data.Dataset.from_tensor_slices(train_dataset)


# def test_mixture_of_svgp_experts_predict(mixture_of_svgp_experts_model):
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#     mixture_of_svgp_experts_model.compile(optimizer=optimizer)
#     gpf.utilities.print_summary(mixture_of_svgp_experts_model)
#     # mixture_dist = mixture_of_svgp_experts_model.build((RegressionData.X.shape[1],))
#     mixture_dist = mixture_of_svgp_experts_model(RegressionData.X)
#     # print(mixture_of_svgp_experts_model.summary())
#     # mixture_dist = mixture_of_svgp_experts_model.predict(RegressionData.X)
#     # mixture_dist = mixture_of_svgp_experts_model.lower_bound_tight(
#     #     train_dataset, num_data=num_data, num_samples=1
#     # )
#     # mixture_dist = mixture_of_svgp_experts_model.lower_bound_further_gating(
#     #     train_dataset, num_data=num_data, num_samples=1
#     # )
#     # mixture_dist = mixture_of_svgp_experts_model.lower_bound_further(
#     #     train_dataset, num_data=num_data, num_samples=1
#     # )
#     mixture_of_svgp_experts_model.fit(
#         train_dataset[0],
#         train_dataset[1],
#         epochs=10,
#         batch_size=3
#         # train_dataset_tf, epochs=10, batch_size=3
#     )
