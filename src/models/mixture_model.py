from abc import ABC, abstractmethod
from typing import Optional, Tuple

import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.models import BayesianModel
from gpflow.training_mixins import InputData, RegressionData

tfd = tfp.distributions


class MixtureModel(BayesianModel, ABC):
    @abstractmethod
    def predict_mixing_probs(self, Xnew):
        raise NotImplementedError

    @abstractmethod
    def predict_component_dists(self, Xnew):
        raise NotImplementedError

    @abstractmethod
    def predict_y(self, Xnew):
        # TODO should this be here
        raise NotImplementedError

    def predict_y_mixture(self, Xnew: InputData) -> tf.Tensor:
        mixing_probs = self.predict_mixing_probs(Xnew)
        dists = self.predict_component_dists(Xnew)
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixing_probs),
            components_distribution=dists)

    def predict_y_samples(self,
                          Xnew: InputData,
                          num_samples: int = 1) -> tf.Tensor:
        return self.predict_y_mixture(Xnew).sample(num_samples)


class MixtureOfExperts(MixtureModel, ABC):
    def __init__(self, gating_network, experts):
        self.gating_network = gating_network
        self.experts = experts

    def predict_mixing_probs(self, Xnew):
        return self.gating_network.predict_mixing_probs(Xnew)

    def predict_component_dists(self, Xnew: InputData) -> tf.Tensor:
        return self.experts.predict_dists(Xnew)


class GPMixtureOfExperts(MixtureOfExperts):
    def __init__(self, gating_network, experts):
        super().__init__(gating_network, experts)

    def maximum_log_likelihood_objective(
            self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Objective for maximum likelihood estimation. Should be maximized. E.g.
        log-marginal likelihood (hyperparameter likelihood) for GPR, or lower
        bound to the log-marginal likelihood (ELBO) for sparse and variational
        GPs.
        """
        X, Y = data

        kl_gating = self.gating_network.prior_kl()
        kls_experts = self.experts.prior_kls()
        kl_experts = tf.reduce_sum(kls_experts)

        num_samples_inducing = 1

        mixing_probs = self.predict_mixing_probs(X)

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
