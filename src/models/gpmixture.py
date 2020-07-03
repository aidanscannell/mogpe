from typing import Tuple

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from experts import ExpertsSeparate, ExpertsShared
from gating_network import GatingNetwork
from gpflow.base import Parameter
from gpflow.ci_utils import ci_niter
from gpflow.config import default_float
from gpflow.models import BayesianModel, ExternalDataTrainingLossMixin
from gpflow.models.model import InputData, MeanAndVariance, RegressionData
from gpflow.models.util import inducingpoint_wrapper
from gpflow.utilities import positive, print_summary, triangular
# from plot_model import plot_and_save
from utils.training import run_adam
from utils.model import init_inducing_variables

tfd = tfp.distributions

from abc import ABC, abstractmethod


class GPMixture(ABC):
    @abstractmethod
    def experts(self):
        raise NotImplementedError


class GPMixture(BayesianModel, ExternalDataTrainingLossMixin):
    def __init__(self,
                 input_dim,
                 output_dim,
                 experts,
                 gating_network,
                 bound='tight',
                 num_data=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bound = bound
        self.num_data = num_data

        self.experts = experts
        self.gating_network = gating_network

    def lower_bound_1(self, X, Y, kl_gating, kl_experts):

        num_samples_inducing = 1

        mixing_probs = self.gating_network.predict_mixing_probs_sample_inducing(
            X, num_samples_inducing)

        # expected_experts = self.experts.experts_expectations_sample_inducing(
        #     X, Y, num_samples_inducing)
        expected_experts = self.experts.experts_expectations(
            X, Y, num_samples_inducing)

        sum_over_indicator = 0
        for expected_expert, mixing_prob in zip(expected_experts,
                                                mixing_probs):
            mixing_prob = tf.reshape(mixing_prob, [-1])
            sum_over_indicator += expected_expert * mixing_prob

        # TODO divide by num inducing point samples
        var_exp = tf.reduce_sum(tf.math.log(sum_over_indicator))

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl_gating.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl_gating.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_gating.dtype)

        return tf.reduce_sum(var_exp) * scale - kl_gating - kl_experts

    def lower_bound_2(self, X, Y, kl_gating, kl_experts):

        mixing_probs = self.gating_network.predict_mixing_probs(X)
        # expected_experts = self.experts.experts_expectations(X, Y)
        expected_experts = self.experts.experts_expectations(
            X, Y, num_samples_inducing=None)

        sum_over_indicator = 0
        for expected_expert, mixing_prob in zip(expected_experts,
                                                mixing_probs):
            sum_over_indicator += expected_expert * mixing_prob

        var_exp = tf.reduce_sum(tf.math.log(sum_over_indicator))
        # var_exp = self._var_expectation(X, Y, num_samples)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl_gating.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl_gating.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_gating.dtype)

        # return tf.reduce_sum(var_exp) * scale
        return tf.reduce_sum(var_exp) * scale - kl_gating - kl_experts

    def lower_bound_3(self, X, Y, kl_gating, kl_experts):
        ''' bound on joint prob p(y, alpha | x) '''
        num_samples_f = 1
        # num_samples_f = None

        # mixing_probs = self.gating_network.predict_mixing_probs(X)
        expected_experts = self.experts.experts_expectations(
            X, Y, num_samples_f)
        var_exp = tf.reduce_sum(tf.math.log(expected_experts))

        # sum_over_indicator = 0
        # for expected_expert, mixing_prob in zip(expected_experts,
        #                                         mixing_probs):
        #     sum_over_indicator += expected_expert * mixing_prob

        # var_exp = tf.reduce_sum(tf.math.log(sum_over_indicator))
        # var_exp = self._var_expectation(X, Y, num_samples)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl_gating.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl_gating.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_gating.dtype)

        return tf.reduce_sum(var_exp) * scale - kl_gating - kl_experts

    def maximum_log_likelihood_objective(self, data: Tuple[tf.Tensor,
                                                           tf.Tensor]):
        X, Y = data
        kl_gating = self.gating_network.prior_kl()
        kls_experts = self.experts.prior_kls()
        kl_experts = tf.reduce_sum(kls_experts)
        if self.bound == 'tight':
            return self.lower_bound_1(X, Y, kl_gating, kl_experts)
        elif self.bound == 'titsias':
            return self.lower_bound_2(X, Y, kl_gating, kl_experts)
        else:
            error_str = "No bound corresponding to " + str(
                self.bound
            ) + " has been implemented. Select either \'tight\' or \'further\'."
            NotImplementedError(error_str)
        # return self.lower_bound_3(X, Y, kl_gating, kl_experts)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This returns the evidence lower bound (ELBO) of the log marginal likelihood.
        """
        return self.maximum_log_likelihood_objective(data)

    def predict_mixing_probs(self, Xnew: InputData):
        """
        Compute the predictive mixing probabilities [P(a=k | Xnew, ...)]^K
        """
        return self.gating_network.predict_mixing_probs(Xnew)

    def predict_gating_h(self,
                         Xnew: InputData,
                         full_cov: bool = False,
                         full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and (co)variance of the gating network GP at the input points Xnew.
        """
        mean_gating, var_gating = self.gating_network.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return mean_gating, var_gating

    def predict_experts_fs(self,
                           Xnew: InputData,
                           full_cov: bool = False,
                           full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and (co)variance of the GPs
        associated with the experts at the input points Xnew.
        """
        f_means, f_vars = self.experts.predict_fs(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return f_means, f_vars

    def predict_experts_ys(self,
                           Xnew: InputData,
                           full_cov: bool = False,
                           full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and (co)variance of the experts (GP+likelihood) at the input points Xnew.
        """
        y_means, y_vars = self.experts.predict_ys(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return y_means, y_vars

    def predict_y_moment_matched(
            self,
            Xnew: InputData,
            full_cov: bool = False,
            full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the moment matched mean and covariance of the held-out data at the input points.
        """
        y_means, y_vars = self.predict_experts_ys(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        # mixing_probs = self.predict_mixing_probs(Xnew)
        y_means = tf.convert_to_tensor(y_means)
        y_vars = tf.convert_to_tensor(y_vars)

        mixing_probs = self.gating_network.predict_mixing_probs_tensor(Xnew)
        mixing_probs = tf.expand_dims(mixing_probs, -1)

        # move mixture dimension to last dimension
        y_means = tf.transpose(y_means, [1, 2, 0])
        y_vars = tf.transpose(y_vars, [1, 2, 0])
        mixing_probs = tf.transpose(mixing_probs, [1, 2, 0])

        gaussian_mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixing_probs),
            components_distribution=tfd.Normal(
                loc=y_means,  # One for each component.
                scale=y_vars))  # And same here.

        y_mean = gaussian_mixture.mean()
        y_var = gaussian_mixture.variance()

        return y_mean, y_var

    def sample_y(self,
                 Xnew: InputData,
                 num_samples=100,
                 full_cov: bool = False,
                 full_output_cov: bool = False) -> MeanAndVariance:
        y_means, y_vars = self.predict_experts_ys(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        # mixing_probs = self.predict_mixing_probs(Xnew)
        y_means = tf.convert_to_tensor(y_means)
        y_vars = tf.convert_to_tensor(y_vars)

        mixing_probs = self.gating_network.predict_mixing_probs_tensor(Xnew)
        mixing_probs = tf.expand_dims(mixing_probs, -1)

        # move mixture dimension to last dimension
        y_means = tf.transpose(y_means, [1, 2, 0])
        y_vars = tf.transpose(y_vars, [1, 2, 0])
        mixing_probs = tf.transpose(mixing_probs, [1, 2, 0])

        gaussian_mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixing_probs),
            components_distribution=tfd.Normal(
                loc=y_means,  # One for each component.
                scale=y_vars))  # And same here.

        return gaussian_mixture.sample(num_samples)


def trim_dataset(X, Y, x1_low=-3., x2_low=-3., x1_high=0., x2_high=-1.):
    mask_0 = X[:, 0] < x1_low
    mask_1 = X[:, 1] < x2_low
    mask_2 = X[:, 0] > x1_high
    mask_3 = X[:, 1] > x2_high
    mask = mask_0 | mask_1 | mask_2 | mask_3
    X_partial = X[mask, :]
    Y_partial = Y[mask, :]
    x1 = [x1_low, x1_low, x1_high, x1_high, x1_low]
    x2 = [x2_low, x2_high, x2_high, x2_low, x2_low]
    X_missing = [x1, x2]

    print("New data shape:", Y_partial.shape)
    return X_partial, Y_partial
