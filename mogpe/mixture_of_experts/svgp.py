#!/usr/bin/env python3
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from gpflow.models import ExternalDataTrainingLossMixin
from gpflow.models.training_mixins import InputData, RegressionData
from mogpe.experts import SVGPExperts
from mogpe.gating_networks import SVGPGatingNetwork
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
        gating_network: SVGPGatingNetwork,
        experts: SVGPExperts,
        num_data: int,
        num_samples: int = 1,
        bound: str = "further",  # "tight" or "further"
    ):
        # assert isinstance(gating_network, SVGPGatingNetworkBase)
        assert isinstance(gating_network, SVGPGatingNetwork)
        assert isinstance(experts, SVGPExperts)
        super().__init__(gating_network, experts)
        self.num_samples = num_samples
        self.num_data = num_data
        self.bound = bound

    def maximum_log_likelihood_objective(
        self, data: Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        if self.bound == "further":
            return self.lower_bound_further(data)
        elif self.bound == "tight":
            return self.lower_bound_tight(data)
        else:
            print(
                'Incorrect value passed so MixtureOfSVGPExperts.bound, should be "tight" or "further". Using further_bound as default.'
            )
            return self.lower_bound_further(data)
        # return self.lower_bound_1(data)
        # return self.lower_bound_dagp(data)
        # return self.lower_bound_analytic(data)

    def lower_bound_further(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Lower bound to the log-marginal likelihood (ELBO).

        Looser bound than lower_bound_1 but analytically marginalises
        the inducing variables $q(\hat{f}, \hat{h})$. Replaces M-dimensional
        approx integrals with 1-dimensional approx integrals.

        This bound assumes each output dimension is independent.

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope("ELBO") as scope:
            X, Y = data

            # Evaluate KL terms
            kl_gating = self.gating_network.prior_kl()
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            # Sample each experts variational posterior q(F) and construct p(Y|F)
            f_means, f_vars = self.experts.predict_fs(X, full_cov=False)
            f_dist = tfd.Normal(loc=f_means, scale=f_vars)  # [N, F, K]
            f_dist_samples = f_dist.sample(self.num_samples)  # [S, N, F, K]
            noise_variances = self.experts.noise_variances()
            components = []
            for expert_k in range(self.num_experts):
                component = tfd.Normal(
                    loc=f_dist_samples[..., expert_k], scale=noise_variances[expert_k]
                )
                components.append(component)

            # Evaluate gating network to get categorical dist over inicator var
            mixing_probs = self.predict_mixing_probs(X)  # [N, K]
            mixing_probs_broadcast = tf.expand_dims(mixing_probs, -2)  # [N, 1, K]
            mixing_probs_broadcast = tf.expand_dims(
                mixing_probs_broadcast, 0
            )  # [1, N, 1, K]
            mixing_probs_broadcast = tf.broadcast_to(
                mixing_probs_broadcast, f_dist_samples.shape  # [S, N, F, K]
            )
            categorical = tfd.Categorical(probs=mixing_probs_broadcast)

            # Create mixture dist and evaluate log prob
            mixture = tfd.Mixture(cat=categorical, components=components)
            variational_expectation = mixture.log_prob(Y)
            print("variational_expectation")
            print(variational_expectation.shape)

            # sum over output dimensions (assumed independent)
            variational_expectation = tf.reduce_sum(variational_expectation, -1)
            print("variational_expectation after sum over output dims")
            print(variational_expectation.shape)

            # average samples (gibbs)
            # TODO have I average gibbs samples correctly???
            approx_variational_expectation = (
                tf.reduce_sum(variational_expectation, axis=0) / self.num_samples
            )
            print("variational_expectation after averaging gibbs samples")
            print(approx_variational_expectation.shape)
            sum_variational_expectation = tf.reduce_sum(
                approx_variational_expectation, axis=0
            )
            print("variational_expectation after sum over data mini batches")
            print(sum_variational_expectation.shape)

            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, default_float())
            return sum_variational_expectation * scale - kl_gating - kl_experts

    def lower_bound_tight(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Lower bound to the log-marginal likelihood (ELBO).

        Tighter bound than lower_bound_further but requires an M dimensional
        expectation over the inducing variables $q(\hat{f}, \hat{h})$
        to be approximated (with Gibbs sampling).

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope("ELBO") as scope:
            X, Y = data
            # num_test = X.shape[0]

            kl_gating = tf.reduce_sum(self.gating_network.prior_kl())
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            with tf.name_scope("predict_mixing_probs") as scope:
                h_mean, h_var = self.gating_network.predict_f(
                    X, num_inducing_samples=self.num_samples, full_cov=False
                )

                def mixing_probs_wrapper(args):
                    mean, var = args
                    return self.gating_network.predict_mixing_probs_given_h(mean, var)

                # map over samples
                mixing_probs = tf.map_fn(
                    mixing_probs_wrapper,
                    (h_mean, h_var),
                    fn_output_signature=default_float(),
                )

                # mixing_probs = tf.expand_dims(mixing_probs, -2)
                print("Mixing probs")
                print(mixing_probs.shape)

            with tf.name_scope("predict_experts_prob") as scope:
                batched_dists = self.predict_experts_dists(
                    X, num_inducing_samples=self.num_samples
                )
                print("batched_dists")
                print(batched_dists)

                Y = tf.expand_dims(Y, 0)
                Y = tf.expand_dims(Y, -1)
                print("Y")
                print(Y.shape)
                expected_experts = batched_dists.prob(Y)
                print("Expected experts")
                print(expected_experts.shape)
                # TODO is it correct to sum over output dimension?
                # sum over output_dim
                expected_experts = tf.reduce_prod(expected_experts, -2)
                print("Experts after product over output dims")
                # print(expected_experts.shape)
                # expected_experts = tf.expand_dims(expected_experts, -2)
                print(expected_experts.shape)

            shape_constraints = [
                (
                    expected_experts,
                    ["num_inducing_samples", "num_data", "num_experts"],
                ),
                (
                    mixing_probs,
                    ["num_inducing_samples", "num_data", "num_experts"],
                ),
            ]
            tf.debugging.assert_shapes(
                shape_constraints,
                message="Gating network and experts dimensions do not match",
            )
            with tf.name_scope("marginalise_indicator_variable") as scope:
                expected_experts = tf.expand_dims(expected_experts, -2)
                expected_experts = tf.expand_dims(expected_experts, 1)
                mixing_probs = tf.expand_dims(mixing_probs, -2)
                mixing_probs = tf.expand_dims(mixing_probs, 0)
                print("expected_experts expanded")
                print("mixing_probs expanded")
                print(expected_experts.shape)
                print(mixing_probs.shape)
                weighted_sum_over_indicator = tf.matmul(
                    expected_experts, mixing_probs, transpose_b=True
                )

                print("Marginalised indicator variable")
                print(weighted_sum_over_indicator.shape)
                # remove last dim as artifacts of marginalising indicator
                weighted_sum_over_indicator = weighted_sum_over_indicator[:, :, :, 0, 0]
                # weighted_sum_over_indicator = weighted_sum_over_indicator[:, :, 0, 0]
                # weighted_sum_over_indicator = weighted_sum_over_indicator[:, :, 0]
            print("Marginalised indicator variable")
            print(weighted_sum_over_indicator.shape)

            # TODO where should output dimension be reduced?
            # weighted_sum_over_indicator = tf.reduce_sum(
            #     weighted_sum_over_indicator, (-2, -1))
            # weighted_sum_over_indicator = tf.reduce_sum(
            #     weighted_sum_over_indicator, (-2, -1))
            # print('Reduce sum over output dimension')
            # print(weighted_sum_over_indicator.shape)

            # # TODO correct num samples for K experts. This assumes 2 experts
            # num_samples = self.num_samples ** (self.num_experts + 1)
            # var_exp = (
            #     1
            #     / num_samples
            #     * tf.reduce_sum(tf.math.log(weighted_sum_over_indicator), axis=0)
            # )
            # print("Averaged inducing samples")
            # print(var_exp.shape)
            # # # TODO where should output dimension be reduced?
            # # var_exp = tf.linalg.diag_part(var_exp)
            # # print('Ignore covariance in output dimension')
            # # print(var_exp.shape)
            # var_exp = tf.reduce_sum(var_exp)
            # print("Reduced sum over mini batch")
            # print(var_exp.shape)

            log = tf.math.log(weighted_sum_over_indicator)
            var_exp = tf.reduce_mean(log, axis=0)  # Average experts inducing samples
            var_exp = tf.reduce_mean(var_exp, axis=0)  # Average gating inducing samples
            print("Averaged inducing samples")
            print(var_exp.shape)
            var_exp = tf.reduce_sum(var_exp, 0)
            print("Reduced sum over mini batch")
            print(var_exp.shape)

            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, default_float())

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
