#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from gpflow.models import BayesianModel, ExternalDataTrainingLossMixin
from gpflow.models.training_mixins import InputData, RegressionData

from mogpe.experts import SVGPExperts
from mogpe.gating_networks import SVGPGatingNetworkBase
from mogpe.mixture_of_experts import MixtureOfExperts

tfd = tfp.distributions


class MixtureOfSVGPExperts(MixtureOfExperts, ExternalDataTrainingLossMixin):
    """Mixture of SVGP experts using stochastic variational inference.

    Implemention of a mixture of Gaussian process (GPs) experts method where
    the gating network is also implemented using GPs.
    The model is trained with stochastic variational inference by exploiting
    the factorization achieved by sparse GPs.

    :param gating_network: an instance of the GatingNetworkBase class with
                            the predict_mixing_probs(Xnew) method implemented.
    :param experts: an instance of the SVGPExperts class with the
                    predict_dists(Xnew) method implemented.
    :param num_inducing_samples: the number of samples to draw from the
                                 inducing point distributions during training.
    :param num_data: the number of data points.
    """
    def __init__(self,
                 gating_network: SVGPGatingNetworkBase,
                 experts: SVGPExperts,
                 num_inducing_samples: int = 1,
                 num_data: int = None):
        assert isinstance(gating_network, SVGPGatingNetworkBase)
        assert isinstance(experts, SVGPExperts)
        super().__init__(gating_network, experts)
        self.num_inducing_samples = num_inducing_samples
        self.num_data = num_data

    def maximum_log_likelihood_objective(self,
                                         data: Tuple[tf.Tensor, tf.Tensor]
                                         ) -> tf.Tensor:
        """Objective for maximum likelihood estimation.

        Lower bound to the log-marginal likelihood (ELBO).

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope('ELBO') as scope:
            X, Y = data
            num_test = X.shape[0]

            # kl_gating = self.gating_network.prior_kl()
            # kls_gatings = self.gating_network.prior_kls()
            kl_gating = tf.reduce_sum(self.gating_network.prior_kls())
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            with tf.name_scope('predict_mixing_probs') as scope:
                mixing_probs = self.predict_mixing_probs(
                    X, num_inducing_samples=self.num_inducing_samples)
            # TODO move this reshape into gating function
            # mixing_probs = tf.reshape(
            #     mixing_probs,
            #     [self.num_inducing_samples, num_test, self.num_experts])
            print("Mixing probs")
            print(mixing_probs.shape)

            with tf.name_scope('predict_experts_prob') as scope:
                batched_dists = self.predict_experts_dists(
                    X, num_inducing_samples=self.num_inducing_samples)

                Y = tf.expand_dims(Y, 0)
                Y = tf.expand_dims(Y, -1)
                expected_experts = batched_dists.prob(Y)
                print('Expected experts')
                print(expected_experts.shape)
                # TODO is it correct to sum over output dimension?
                # sum over output_dim
                expected_experts = tf.reduce_prod(expected_experts, -2)
                print('Experts after product over output dims')
                # print(expected_experts.shape)
                expected_experts = tf.expand_dims(expected_experts, -2)
                print(expected_experts.shape)

            shape_constraints = [
                (expected_experts,
                 ["num_inducing_samples", "num_data", "1", "num_experts"]),
                (mixing_probs,
                 ["num_inducing_samples", "num_data", "1", "num_experts"]),
            ]
            tf.debugging.assert_shapes(
                shape_constraints,
                message="Gating network and experts dimensions do not match")
            with tf.name_scope('marginalise_indicator_variable') as scope:
                weighted_sum_over_indicator = tf.matmul(expected_experts,
                                                        mixing_probs,
                                                        transpose_b=True)

                # remove last two dims as artifacts of marginalising indicator
                weighted_sum_over_indicator = weighted_sum_over_indicator[:, :, 0, 0]
            print('Marginalised indicator variable')
            print(weighted_sum_over_indicator.shape)

            # TODO where should output dimension be reduced?
            # weighted_sum_over_indicator = tf.reduce_sum(
            #     weighted_sum_over_indicator, (-2, -1))
            # weighted_sum_over_indicator = tf.reduce_sum(
            #     weighted_sum_over_indicator, (-2, -1))
            # print('Reduce sum over output dimension')
            # print(weighted_sum_over_indicator.shape)

            # TODO correct num samples for K experts. This assumes 2 experts
            num_samples = self.num_inducing_samples**(self.num_experts + 1)
            var_exp = 1 / num_samples * tf.reduce_sum(
                tf.math.log(weighted_sum_over_indicator), axis=0)
            print('Averaged inducing samples')
            print(var_exp.shape)
            # # TODO where should output dimension be reduced?
            # var_exp = tf.linalg.diag_part(var_exp)
            # print('Ignore covariance in output dimension')
            # print(var_exp.shape)
            var_exp = tf.reduce_sum(var_exp)
            print('Reduced sum over mini batch')
            print(var_exp.shape)

            # var_exp = tf.reduce_sum(var_exp)
            # print('Reduce sum over output_dim to get loss')
            # print(var_exp.shape)

            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, default_float())

            return var_exp * scale - kl_gating - kl_experts

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """Returns the evidence lower bound (ELBO) of the log marginal likelihood.

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        """
        return self.maximum_log_likelihood_objective(data)

    def predict_experts_fs(self,
                           Xnew: InputData,
                           num_inducing_samples: int = None,
                           full_cov=False,
                           full_output_cov=False
                           ) -> Tuple[tf.Tensor, tf.Tensor]:
        """"Compute mean and (co)variance of experts latent functions at Xnew.

        If num_inducing_samples is not None then sample inducing points instead
        of analytically integrating them. This is required in the mixture of
        experts lower bound.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param num_inducing_samples: the number of samples to draw from the
                                    inducing point distributions during training.
        :returns: a tuple of (mean, (co)var) each with shape [..., num_test, output_dim, num_experts]
        """
        return self.experts.predict_fs(Xnew, num_inducing_samples, full_cov,
                                       full_output_cov)
