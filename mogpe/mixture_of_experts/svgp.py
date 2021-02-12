#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from gpflow.models import ExternalDataTrainingLossMixin
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

    def __init__(
        self,
        gating_network: SVGPGatingNetworkBase,
        experts: SVGPExperts,
        num_data: int,
        num_inducing_samples: int = 0,
    ):
        assert isinstance(gating_network, SVGPGatingNetworkBase)
        assert isinstance(experts, SVGPExperts)
        super().__init__(gating_network, experts)
        self.num_inducing_samples = num_inducing_samples
        self.num_data = num_data

    def maximum_log_likelihood_objective(
        self, data: Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        # return self.lower_bound_analytic_simple(data)
        # return self.lower_bound_analytic(data)
        return self.lower_bound_analytic_2(data)
        # return self.lower_bound_stochastic(data)
        # return self.lower_bound_dagp(data)

    def lower_bound_analytic_simple(
        self, data: Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        """Lower bound to the log-marginal likelihood (ELBO).

        This bound calculates log p(y_d|x) (for each output dimension) and sums
        over them outside of the log to achieve independence.

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope("ELBO") as scope:
            X, Y = data
            num_test = X.shape[0]

            kl_gating = tf.reduce_sum(self.gating_network.prior_kls())
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            print("Y shape")
            print(Y.shape)
            # Y = tf.expand_dims(Y, -1)
            # print(Y.shape)
            y_dist = self.predict_y(X)
            print("y_dist (expert indicator variable marginalised)")
            print(y_dist.batch_shape)
            print(y_dist)

            log_prob_y = y_dist.log_prob(Y)
            print("log_prob_y shape")
            print(log_prob_y.shape)

            # Assume output dimensions are independent so sum over them
            # as we've moved outside of the log
            var_exp = tf.reduce_sum(log_prob_y, -1)
            print("Shape after reducing sum over output dimension")
            print(var_exp.shape)

            var_exp = tf.reduce_sum(var_exp)
            print("Reduced sum over mini batch")
            print(var_exp.shape)
            print(var_exp)

            print("num data")
            print(self.num_data)
            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                print("minibatch_size")
                print(minibatch_size)
                scale = num_data / minibatch_size
                print(scale)
            else:
                scale = tf.cast(1.0, default_float())
            print("kl")
            print(kl_gating)
            print(kl_experts)

            return var_exp * scale - kl_gating - kl_experts

    def lower_bound_dagp(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Lower bound used in Data Association with GPs (DAGP).

        This bound doesn't marginalise the expert indicator variable.

        TODO check I've implemented this correctlyy. It's definitely slower thatn it should be.

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope("ELBO") as scope:
            X, Y = data
            Y = tf.expand_dims(Y, -1)
            num_test = X.shape[0]

            kl_gating = tf.reduce_sum(self.gating_network.prior_kls())
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            h_means, h_vars = self.gating_network.predict_fs(X)
            # prob_a_0 = self.likelihood.predict_mean_and_var(h_means, h_var)[0]
            # TODO remove duplicate call of h_means, h_vars

            mixing_probs = self.predict_mixing_probs(X)
            print("Mixing probs")
            print(mixing_probs.shape)
            assignments = tf.where(
                mixing_probs > 0.5,
                tf.ones(mixing_probs.shape),
                tf.zeros(mixing_probs.shape),
            )
            print("Assignments")
            print(assignments.shape)

            log_expected_gating_fs = (
                self.gating_network.likelihood.predict_log_density(
                    h_means[:, 0, :], h_vars[:, 0, :], assignments[:, 0, :]
                )
                # self.gating_network.likelihood.predict_log_density(
                #     h_means[:, :, 0], h_vars[:, :, 0], assignments[:, :, 0]
                # )
            )
            print("log expected gating fs")
            print(log_expected_gating_fs.shape)
            var_exp_gating_fs = tf.reduce_sum(log_expected_gating_fs)
            print(var_exp_gating_fs.shape)

            batched_dists = self.predict_experts_dists(X)
            print("Experts dists")
            print(batched_dists)

            log_expected_experts = batched_dists.log_prob(Y)
            print("Log expected experts")
            print(log_expected_experts.shape)

            log_expected_experts = tf.reduce_sum(log_expected_experts, -2)
            print("Experts after sum over output dims")
            print(log_expected_experts.shape)

            # TODO this only works for 2 experts case
            var_exp_experts = tf.where(
                assignments[:, 0, 0] == 1,
                log_expected_experts[:, 0],
                log_expected_experts[:, 1],
            )
            print("Experts after selecting mode")
            print(var_exp_experts.shape)

            var_exp_experts = tf.reduce_sum(var_exp_experts)
            print("Reduced sum over mini batch")
            print(var_exp_experts.shape)

            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, default_float())

            return (
                var_exp_gating_fs * scale
                + var_exp_experts * scale
                - kl_gating
                - kl_experts
            )

    def lower_bound_analytic(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Lower bound to the log-marginal likelihood (ELBO).

        This bound assumes each output dimension is independent and takes
        the product over them within the logarithm (and before the expert
        indicator variable is marginalised).

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope("ELBO") as scope:
            X, Y = data
            num_test = X.shape[0]

            # kl_gating = self.gating_network.prior_kl()
            # kls_gatings = self.gating_network.prior_kls()
            kl_gating = tf.reduce_sum(self.gating_network.prior_kls())
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            mixing_probs = self.predict_mixing_probs(X)
            print("Mixing probs")
            print(mixing_probs.shape)

            batched_dists = self.predict_experts_dists(X)

            Y = tf.expand_dims(Y, -1)
            expected_experts = batched_dists.prob(Y)
            print("Expected experts")
            print(expected_experts.shape)
            # product over output dimensions (assumed independent)
            expected_experts = tf.reduce_prod(expected_experts, -2)
            print("Experts after product over output dims")
            # print(expected_experts.shape)
            expected_experts = tf.expand_dims(expected_experts, -2)
            print(expected_experts.shape)

            shape_constraints = [
                (expected_experts, ["num_data", "1", "num_experts"]),
                (mixing_probs, ["num_data", "1", "num_experts"]),
            ]
            tf.debugging.assert_shapes(
                shape_constraints,
                message="Gating network and experts dimensions do not match",
            )
            with tf.name_scope("marginalise_indicator_variable") as scope:
                weighted_sum_over_indicator = tf.matmul(
                    expected_experts, mixing_probs, transpose_b=True
                )

                # remove last two dims as artifacts of marginalising indicator
                weighted_sum_over_indicator = weighted_sum_over_indicator[:, 0, 0]
            print("Marginalised indicator variable")
            print(weighted_sum_over_indicator.shape)

            # TODO where should output dimension be reduced?

            # TODO correct num samples for K experts. This assumes 2 experts
            # num_samples = self.num_inducing_samples**(self.num_experts + 1)
            var_exp = tf.reduce_sum(tf.math.log(weighted_sum_over_indicator), axis=0)
            print("Reduced sum over mini batch")
            print(var_exp.shape)

            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, default_float())
            return var_exp * scale - kl_gating - kl_experts

    def lower_bound_analytic_2(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Lower bound to the log-marginal likelihood (ELBO).

        This bound assumes each output dimension is independent and takes
        the product over them within the logarithm (and before the expert
        indicator variable is marginalised).

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope("ELBO") as scope:
            X, Y = data
            num_test = X.shape[0]

            # kl_gating = self.gating_network.prior_kl()
            # kls_gatings = self.gating_network.prior_kls()
            kl_gating = tf.reduce_sum(self.gating_network.prior_kls())
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            mixing_probs = self.predict_mixing_probs(X)
            print("Mixing probs")
            print(mixing_probs.shape)

            # likelihoods = self.experts.likelihoods()
            # print("likelihoods")
            # print(likelihoods)
            noise_variances = self.experts.noise_variances()
            print("noise_variances")
            print(noise_variances)

            fmeans, fvars = self.experts.predict_fs(X, full_cov=False)
            print("here")
            print(fmeans.shape)
            print(fvars.shape)
            f_dist = tfp.distributions.Normal(
                loc=fmeans,
                scale=fvars,
            )
            print("f_dist")
            print(f_dist)
            # num_samples = 5
            num_samples = 1
            f_dist_samples = f_dist.sample(num_samples)
            print(f_dist_samples.shape)

            components = []
            for expert_k in range(self.num_experts):
                locs = f_dist_samples[..., expert_k]
                print("locs")
                print(locs.shape)
                component = tfd.Normal(
                    loc=f_dist_samples[..., expert_k], scale=noise_variances[expert_k]
                )
                print("component")
                print(component)
                components.append(component)
                # mixing_probs_list = mixing_probs[..., expert_k]
                # print("mixing_probs_list")
                # print(mixing_probs_list)
            print("components")
            print(components)
            mixing_probs_broadcast = tf.expand_dims(mixing_probs, 0)
            mixing_probs_broadcast = tf.broadcast_to(
                mixing_probs_broadcast, f_dist_samples.shape
            )
            print("mixing_probs_broadcast")
            print(mixing_probs_broadcast)
            categorical = tfd.Categorical(probs=mixing_probs_broadcast)
            print("cat")
            print(categorical)
            mixture = tfd.Mixture(cat=categorical, components=components)
            print("mixture")
            print(mixture)
            variational_expectation = mixture.log_prob(Y)
            print("variational_expectation")
            print(variational_expectation)

            # sum over output dimensions (assumed independent)
            variational_expectation = tf.reduce_sum(variational_expectation, -1)
            print("variational_expectation after sum over output dims")
            print(variational_expectation)

            # average samples (gibbs)
            # TODO have I average gibbs samples correctly???
            approx_variational_expectation = (
                tf.reduce_sum(variational_expectation, axis=0) / num_samples
            )
            print("variational_expectation after averaging gibbs samples")
            print(approx_variational_expectation)
            sum_variational_expectation = tf.reduce_sum(
                approx_variational_expectation, axis=0
            )
            print("variational_expectation after sum over data mini batches")
            print(sum_variational_expectation)

            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, default_float())
            return sum_variational_expectation * scale - kl_gating - kl_experts

    def lower_bound_stochastic(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Objective for maximum likelihood estimation.

        Lower bound to the log-marginal likelihood (ELBO).

        Same as lower_bound_analytic except that the inducing point dists
        q(f, h) are marginalised via Gibbs sampling.

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope("ELBO") as scope:
            X, Y = data
            num_test = X.shape[0]

            # kl_gating = self.gating_network.prior_kl()
            # kls_gatings = self.gating_network.prior_kls()
            kl_gating = tf.reduce_sum(self.gating_network.prior_kls())
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            with tf.name_scope("predict_mixing_probs") as scope:
                mixing_probs = self.predict_mixing_probs(
                    X, num_inducing_samples=self.num_inducing_samples
                )
            # TODO move this reshape into gating function
            # mixing_probs = tf.reshape(
            #     mixing_probs,
            #     [self.num_inducing_samples, num_test, self.num_experts])
            print("Mixing probs")
            print(mixing_probs.shape)

            with tf.name_scope("predict_experts_prob") as scope:
                batched_dists = self.predict_experts_dists(
                    X, num_inducing_samples=self.num_inducing_samples
                )

                Y = tf.expand_dims(Y, 0)
                Y = tf.expand_dims(Y, -1)
                expected_experts = batched_dists.prob(Y)
                print("Expected experts")
                print(expected_experts.shape)
                # TODO is it correct to sum over output dimension?
                # sum over output_dim
                expected_experts = tf.reduce_prod(expected_experts, -2)
                print("Experts after product over output dims")
                # print(expected_experts.shape)
                expected_experts = tf.expand_dims(expected_experts, -2)
                print(expected_experts.shape)

            shape_constraints = [
                (
                    expected_experts,
                    ["num_inducing_samples", "num_data", "1", "num_experts"],
                ),
                (
                    mixing_probs,
                    ["num_inducing_samples", "num_data", "1", "num_experts"],
                ),
            ]
            tf.debugging.assert_shapes(
                shape_constraints,
                message="Gating network and experts dimensions do not match",
            )
            with tf.name_scope("marginalise_indicator_variable") as scope:
                weighted_sum_over_indicator = tf.matmul(
                    expected_experts, mixing_probs, transpose_b=True
                )

                # remove last two dims as artifacts of marginalising indicator
                weighted_sum_over_indicator = weighted_sum_over_indicator[:, :, 0, 0]
            print("Marginalised indicator variable")
            print(weighted_sum_over_indicator.shape)

            # TODO where should output dimension be reduced?
            # weighted_sum_over_indicator = tf.reduce_sum(
            #     weighted_sum_over_indicator, (-2, -1))
            # weighted_sum_over_indicator = tf.reduce_sum(
            #     weighted_sum_over_indicator, (-2, -1))
            # print('Reduce sum over output dimension')
            # print(weighted_sum_over_indicator.shape)

            # TODO correct num samples for K experts. This assumes 2 experts
            num_samples = self.num_inducing_samples ** (self.num_experts + 1)
            var_exp = (
                1
                / num_samples
                * tf.reduce_sum(tf.math.log(weighted_sum_over_indicator), axis=0)
            )
            print("Averaged inducing samples")
            print(var_exp.shape)
            # # TODO where should output dimension be reduced?
            # var_exp = tf.linalg.diag_part(var_exp)
            # print('Ignore covariance in output dimension')
            # print(var_exp.shape)
            var_exp = tf.reduce_sum(var_exp)
            print("Reduced sum over mini batch")
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
        # return self.maximum_log_likelihood_objective_stochastic(data)
        return self.maximum_log_likelihood_objective(data)

    def predict_experts_fs(
        self,
        Xnew: InputData,
        num_inducing_samples: int = None,
        full_cov=False,
        full_output_cov=False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """ "Compute mean and (co)variance of experts latent functions at Xnew.

        If num_inducing_samples is not None then sample inducing points instead
        of analytically integrating them. This is required in the mixture of
        experts lower bound.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param num_inducing_samples: the number of samples to draw from the
                                    inducing point distributions during training.
        :returns: a tuple of (mean, (co)var) each with shape [..., num_test, output_dim, num_experts]
        """
        return self.experts.predict_fs(
            Xnew, num_inducing_samples, full_cov, full_output_cov
        )
