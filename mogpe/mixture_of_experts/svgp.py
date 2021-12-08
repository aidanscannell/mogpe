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
        bound: str = "further_gating",  # "further_gating" or "tight" or "further" or
    ):
        assert isinstance(gating_network, SVGPGatingNetwork)
        assert isinstance(experts, SVGPExperts)
        super().__init__(gating_network, experts)
        self.num_samples = num_samples
        self.num_data = num_data
        self.bound = bound

    def maximum_log_likelihood_objective(
        self, data: Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        # self.marginal_likelihood_new(data)
        # return self.lower_bound_tight(data)
        # return self.lower_bound_further_experts(data)
        # return self.lower_bound_further_2(data)
        # return self.lower_bound_tight_2(data)
        if self.bound == "further":
            return self.lower_bound_further(data)
        elif self.bound == "tight":
            return self.lower_bound_tight(data)
        elif self.bound == "tight_2":
            return self.lower_bound_tight_2(data)
        elif self.bound == "further_gating":
            return self.lower_bound_further_gating(data)
        elif self.bound == "further_expert":
            return self.lower_bound_further_experts(data)
        else:
            print(
                'Incorrect value passed so MixtureOfSVGPExperts.bound, should be "tight" or "further_gating" or "further". Using further_gating as default.'
            )
            return self.lower_bound_tight(data)
        # return self.lower_bound_1(data)
        # return self.lower_bound_dagp(data)
        # return self.lower_bound_analytic(data)

    def lower_bound_further_gating(
        self, data: Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        """Lower bound to the log-marginal likelihood (ELBO).

        Similar to lower_bound_tight but with a further bound on the gating
        network. The bound removes the M dimensional integral over the gating
        network inducing variables $q(\hat{\mathbf{U}})$ with 1 dimensional
        integrals over the gating network variational posterior $q(\mathbf{h}_n)$.

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope("ELBO") as scope:
            X, Y = data

            kl_gating = tf.reduce_sum(self.gating_network.prior_kl())
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            with tf.name_scope("predict_mixing_probs") as scope:
                # Evaluate gating network to get categorical dist over inicator var
                h_means, h_vars = self.gating_network.predict_f(X, full_cov=False)
                h_dist = tfd.Normal(loc=h_means, scale=h_vars)  # [N, F, K]
                h_dist_samples = h_dist.sample(self.num_samples)  # [S, N, K]
                mixing_probs = self.gating_network.predict_mixing_probs_given_h(
                    h_mean=h_dist_samples
                )  # [S, N, K]
                print("Mixing probs")
                print(mixing_probs.shape)

            with tf.name_scope("predict_experts_prob") as scope:
                batched_dists = self.predict_experts_dists(
                    X, num_inducing_samples=self.num_samples
                )  # [S, N, F, K]
                print("batched_dists")
                print(batched_dists.batch_shape)
                Y = tf.expand_dims(Y, 0)  # [1, N, F]
                Y = tf.expand_dims(Y, -1)  # [1, N, F, 1]
                expected_experts = batched_dists.prob(Y)  # [S, N, F, K]
                print("Expected experts")
                print(expected_experts.shape)

                # Product over output_dim
                expected_experts = tf.reduce_prod(expected_experts, -2)  # [S, N, K]
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

            # Expand to enable integrationg over both expert and gating samples
            expected_experts = tf.expand_dims(expected_experts, 1)
            mixing_probs = tf.expand_dims(mixing_probs, 0)

            with tf.name_scope("marginalise_indicator_variable") as scope:
                expected_experts = tf.expand_dims(expected_experts, -2)
                mixing_probs = tf.expand_dims(mixing_probs, -2)
                print("expected_experts expanded")
                print("mixing_probs expanded")
                print(expected_experts.shape)
                print(mixing_probs.shape)
                weighted_sum_over_indicator = tf.matmul(
                    expected_experts, mixing_probs, transpose_b=True
                )
                print("Marginalised indicator variable")
                print(weighted_sum_over_indicator.shape)
                weighted_sum_over_indicator = weighted_sum_over_indicator[:, :, :, 0, 0]
                print(weighted_sum_over_indicator.shape)

            log = tf.math.log(weighted_sum_over_indicator)
            var_exp = tf.reduce_mean(log, axis=0)  # Average gating samples
            var_exp = tf.reduce_mean(var_exp, axis=0)  # Average expert inducing samples
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

    def lower_bound_further(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Lower bound to the log-marginal likelihood (ELBO).

        Looser bound than lower_bound_tight as it marginalises both of the expert's
        and the gating network's inducing variables $q(\hat{f}, \hat{h})$ in closed-form.
        Replaces M-dimensional approx integrals with 1-dimensional approx integrals.

        This bound is equivalent to a different likelihood approximation that
        only mixes the noise models (as opposed to the full GPs).

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
            h_means, h_vars = self.gating_network.predict_f(X, full_cov=False)
            h_dist = tfd.Normal(loc=h_means, scale=h_vars)  # [N, F, K]
            h_dist_samples = h_dist.sample(self.num_samples)  # [S, N, F, K]
            mixing_probs = self.gating_network.predict_mixing_probs_given_h(
                h_mean=h_dist_samples
            )  # [S, N, K]
            mixing_probs_broadcast = tf.expand_dims(mixing_probs, -2)  # [S, N, 1, K]
            mixing_probs_broadcast = tf.broadcast_to(
                mixing_probs_broadcast, f_dist_samples.shape  # [S, N, F, K]
            )
            categorical = tfd.Categorical(probs=mixing_probs_broadcast)

            # Move both expectations inside log
            # mixing_probs = self.predict_mixing_probs(X)  # [N, K]
            # mixing_probs_broadcast = tf.expand_dims(mixing_probs, -2)  # [N, 1, K]
            # mixing_probs_broadcast = tf.broadcast_to(
            #     mixing_probs_broadcast, f_dist.batch_shape  # [S, N, F, K]
            # )
            # categorical = tfd.Categorical(probs=mixing_probs_broadcast)

            # Move gating expectation inside log
            # mixing_probs = self.predict_mixing_probs(X)  # [N, K]
            # mixing_probs_broadcast = tf.expand_dims(mixing_probs, -2)  # [N, 1, K]
            # mixing_probs_broadcast = tf.expand_dims(
            #     mixing_probs_broadcast, 0
            # )  # [1, N, 1, K]
            # mixing_probs_broadcast = tf.broadcast_to(
            #     mixing_probs_broadcast, f_dist_samples.shape  # [S, N, F, K]
            # )
            # categorical = tfd.Categorical(probs=mixing_probs_broadcast)

            # Create mixture dist and evaluate log prob
            mixture = tfd.Mixture(cat=categorical, components=components)
            variational_expectation = mixture.log_prob(Y)  # [S, N, F]
            print("variational_expectation")
            print(variational_expectation.shape)

            # sum over output dimensions (assumed independent)
            variational_expectation = tf.reduce_sum(
                variational_expectation, -1
            )  # [S, N]
            print("variational_expectation after sum over output dims")
            print(variational_expectation.shape)

            # Average samples (gibbs)
            approx_variational_expectation = tf.reduce_mean(
                variational_expectation, axis=0
            )  # [N]
            print("variational_expectation after averaging gibbs samples")
            print(approx_variational_expectation.shape)
            sum_variational_expectation = tf.reduce_sum(
                approx_variational_expectation, axis=0
            )  # []
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

    def lower_bound_further_2(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Lower bound to the log-marginal likelihood (ELBO).

        Looser bound than lower_bound_tight but marginalises the inducing variables
        $q(\hat{f}, \hat{h})$ in closed-form. Replaces M-dimensional
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

            # Evaluate gating network to get categorical dist over inicator var
            h_means, h_vars = self.gating_network.predict_f(X, full_cov=False)
            h_dist = tfd.Normal(loc=h_means, scale=h_vars)  # [N, F, K]
            h_dist_samples = h_dist.sample(self.num_samples)  # [S, N, F, K]
            mixing_probs = self.gating_network.predict_mixing_probs_given_h(
                h_mean=h_dist_samples
            )  # [S, N, K]

            # Sample each experts variational posterior q(F) and construct p(Y|F)
            f_means, f_vars = self.experts.predict_fs(X, full_cov=False)
            f_dist = tfd.Normal(loc=f_means, scale=f_vars)  # [N, F, K]
            f_dist_samples = f_dist.sample(self.num_samples)  # [S, N, F, K]
            noise_variances = self.experts.noise_variances()

            noise_variances = tf.stack(noise_variances, -1)
            noise_variances = tf.expand_dims(noise_variances, 0)
            noise_variances = tf.expand_dims(noise_variances, 0)
            print("noise_variances")
            print(noise_variances)
            batched_dists = tfd.Normal(loc=f_dist_samples, scale=noise_variances)
            print("batched_dists")
            print(batched_dists)

            Y = tf.expand_dims(Y, 0)
            Y = tf.expand_dims(Y, -1)
            Y = tf.broadcast_to(Y, batched_dists.batch_shape)
            print("Y")
            print(Y.shape)
            experts_prob_ys = batched_dists.prob(Y)
            print("Experts probs")
            print(experts_prob_ys.shape)

            # Product over output_dim
            experts_prob_ys = tf.reduce_prod(experts_prob_ys, -2)

            with tf.name_scope("marginalise_indicator_variable") as scope:
                experts_prob_ys = tf.expand_dims(experts_prob_ys, 1)
                mixing_probs = tf.expand_dims(mixing_probs, 0)
                experts_prob_ys = tf.expand_dims(experts_prob_ys, -2)
                mixing_probs = tf.expand_dims(mixing_probs, -2)
                print("expected_experts expanded")
                print(experts_prob_ys.shape)
                print("mixing_probs expanded")
                print(mixing_probs.shape)
                weighted_sum_over_indicator = tf.matmul(
                    experts_prob_ys, mixing_probs, transpose_b=True
                )[..., 0, 0]

                print("Marginalised indicator variable")
                print(weighted_sum_over_indicator.shape)

            log = tf.math.log(weighted_sum_over_indicator)
            var_exp = tf.reduce_mean(log, axis=0)  # Average gating samples
            var_exp = tf.reduce_mean(var_exp, axis=0)  # Average expert inducing samples
            print("Averaged inducing samples")
            print(var_exp.shape)
            sum_variational_expectation = tf.reduce_sum(var_exp, 0)
            print("Reduced sum over mini batch")
            print(sum_variational_expectation.shape)

            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, default_float())
            return sum_variational_expectation * scale - kl_gating - kl_experts

    def lower_bound_tight_2(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
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
                )  # [S, N, K]

                # mixing_probs = tf.expand_dims(mixing_probs, -2)
                print("Mixing probs")
                print(mixing_probs.shape)

            with tf.name_scope("predict_experts_prob") as scope:
                batched_dists = self.predict_experts_dists(
                    X, num_inducing_samples=self.num_samples
                )  # [S, N, F, K]
                print("batched_dists")
                print(batched_dists)

            # components = []
            y_means = batched_dists.mean()
            y_vars = batched_dists.variance()
            print("y_means")
            print(y_means)
            print(y_vars)
            components = []
            for expert_k in range(self.num_experts):
                component = tfd.Normal(
                    loc=y_means[..., expert_k], scale=y_vars[..., expert_k]
                )
                components.append(component)

            mixing_probs_broadcast = tf.expand_dims(mixing_probs, -2)  # [S, N, 1, K]
            mixing_probs_broadcast = tf.broadcast_to(
                mixing_probs_broadcast, batched_dists.batch_shape  # [S, N, F, K]
            )
            categorical = tfd.Categorical(probs=mixing_probs_broadcast)
            print("categorical")
            print(components)
            print(categorical)

            # Create mixture dist and evaluate log prob
            mixture = tfd.Mixture(cat=categorical, components=components)
            print("mixture")
            print(mixture)
            variational_expectation = mixture.log_prob(Y)  # [S, N, F]
            print("variational_expectation")
            print(variational_expectation)

            var_exp = tf.reduce_sum(
                variational_expectation, axis=-1
            )  # Sum over output dims
            # TODO average gating/expert samples separately
            var_exp = tf.reduce_mean(
                var_exp, axis=0
            )  # Average experts inducing samples
            # var_exp = tf.reduce_mean(var_exp, axis=0)  # Average gating inducing samples
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

    def lower_bound_further_experts(
        self, data: Tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        """Lower bound to the log-marginal likelihood (ELBO).

        Similar to lower_bound_tight but with a further bound on the experts.
        The bound removes the M dimensional integral over each expert's
        inducing variables $q(\hat{\mathbf{U}})$ with 1 dimensional
        integrals over the gating network variational posterior $q(\mathbf{h}_n)$.

        This bound is equivalent to a different likelihood approximation that
        only mixes the noise models (as opposed to the full GPs).

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        print("inside further_experts bound")
        with tf.name_scope("ELBO") as scope:
            X, Y = data

            kl_gating = tf.reduce_sum(self.gating_network.prior_kl())
            kl_experts = tf.reduce_sum(self.experts.prior_kls())

            # with tf.name_scope("predict_mixing_probs") as scope:
            #     # Evaluate gating network to get categorical dist over inicator var
            #     h_means, h_vars = self.gating_network.predict_f(X, full_cov=False)
            #     h_dist = tfd.Normal(loc=h_means, scale=h_vars)  # [N, F, K]
            #     h_dist_samples = h_dist.sample(self.num_samples)  # [S, N, K]
            #     mixing_probs = self.gating_network.predict_mixing_probs_given_h(
            #         h_mean=h_dist_samples
            #     )  # [S, N, K]
            #     print("Mixing probs")
            #     print(mixing_probs.shape)

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
                )  # [S, N, K]

                # mixing_probs = tf.expand_dims(mixing_probs, -2)
                print("Mixing probs")
                print(mixing_probs.shape)

                # Get categorical dist over inicator var
                mixing_probs_broadcast = tf.expand_dims(
                    mixing_probs, -2
                )  # [S, N, 1, K]
                mixing_probs_broadcast = tf.broadcast_to(
                    mixing_probs_broadcast, f_dist_samples.shape  # [S, N, F, K]
                )
                categorical = tfd.Categorical(probs=mixing_probs_broadcast)

            # Create mixture dist and evaluate log prob
            mixture = tfd.Mixture(cat=categorical, components=components)
            variational_expectation = mixture.log_prob(Y)  # [S, N, F]
            print("variational_expectation")
            print(variational_expectation.shape)

            # sum over output dimensions (assumed independent)
            variational_expectation = tf.reduce_sum(
                variational_expectation, -1
            )  # [S, N]
            print("variational_expectation after sum over output dims")
            print(variational_expectation.shape)

            # Average samples (gibbs)
            approx_variational_expectation = tf.reduce_mean(
                variational_expectation, axis=0
            )  # [N]
            print("variational_expectation after averaging gibbs samples")
            print(approx_variational_expectation.shape)
            sum_variational_expectation = tf.reduce_sum(
                approx_variational_expectation, axis=0
            )  # []
            print("variational_expectation after sum over data mini batches")
            print(sum_variational_expectation.shape)

            # # Sample each experts variational posterior q(F) and construct p(Y|F)
            # f_means, f_vars = self.experts.predict_fs(X, full_cov=False)
            # tf.print("f_means")
            # tf.print(f_means)
            # tf.print(f_vars)
            # f_dist = tfd.Normal(loc=f_means, scale=f_vars)  # [N, F, K]
            # f_dist_samples = f_dist.sample(self.num_samples)  # [S, N, F, K]
            # noise_variances = self.experts.noise_variances()

            # noise_variances = tf.stack(noise_variances, -1)
            # noise_variances = tf.expand_dims(noise_variances, 0)
            # noise_variances = tf.expand_dims(noise_variances, 0)
            # print("noise_variances")
            # print(noise_variances)
            # batched_dists = tfd.Normal(loc=f_dist_samples, scale=noise_variances)
            # print("batched_dists")
            # print(batched_dists)

            # Y = tf.expand_dims(Y, 0)
            # Y = tf.expand_dims(Y, -1)
            # # Y = tf.broadcast_to(Y, batched_dists.batch_shape)
            # print("Y")
            # print(Y.shape)
            # experts_prob_ys = batched_dists.prob(Y)
            # print("Experts probs")
            # print(experts_prob_ys.shape)
            # tf.print("Experts probs")
            # tf.print(experts_prob_ys)

            # # Product over output_dim
            # experts_prob_ys = tf.reduce_prod(experts_prob_ys, -2)
            # print("Experts after product over output dims")
            # print(experts_prob_ys)

            # with tf.name_scope("marginalise_indicator_variable") as scope:
            #     experts_prob_ys = tf.expand_dims(experts_prob_ys, 1)
            #     mixing_probs = tf.expand_dims(mixing_probs, 0)
            #     experts_prob_ys = tf.expand_dims(experts_prob_ys, -2)
            #     mixing_probs = tf.expand_dims(mixing_probs, -2)
            #     print("expected_experts expanded")
            #     print(experts_prob_ys.shape)
            #     print("mixing_probs expanded")
            #     print(mixing_probs.shape)
            #     weighted_sum_over_indicator = tf.matmul(
            #         experts_prob_ys, mixing_probs, transpose_b=True
            #     )[..., 0, 0]

            #     print("Marginalised indicator variable")
            #     print(weighted_sum_over_indicator.shape)
            #     tf.print("Marginalised indicator variable")
            #     tf.print(weighted_sum_over_indicator)

            # log = tf.math.log(weighted_sum_over_indicator)
            # tf.print("log")
            # tf.print(log)
            # var_exp = tf.reduce_mean(log, axis=0)  # Average gating samples
            # var_exp = tf.reduce_mean(var_exp, axis=0)  # Average expert inducing samples
            # print("Averaged inducing samples")
            # print(var_exp.shape)
            # var_exp = tf.reduce_sum(var_exp, 0)
            # print("Reduced sum over mini batch")
            # print(var_exp.shape)

            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, default_float())

            return sum_variational_expectation * scale - kl_gating - kl_experts
            # return var_exp * scale - kl_gating - kl_experts

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

    def marginal_likelihood_new(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Marginal likelihood (ML).

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: marginal likelihood - a Tensor with shape ()
        """
        X, Y = data
        # mixture_dist = self.predict_y(X)

        mixing_probs = self.predict_mixing_probs(X)
        tf.print("mixing_probs")
        tf.print(mixing_probs)
        multinomial = tfp.distributions.Multinomial(
            # self.num_experts,
            X.shape[0],
            probs=tf.transpose(mixing_probs),
        )
        tf.print("multinomial.prob")
        # ones = tf.ones(self.num_experts, dtype=default_float())
        ones = tf.ones(X.shape[0], dtype=default_float())
        counts = [0 * ones, 1 * ones, 3 * ones]
        # multinomial_prob = multinomial.prob(0 * ones)
        multinomial_prob = multinomial.prob(counts)
        tf.print(multinomial_prob)
        multinomial_prob = multinomial.prob(1 * ones)
        tf.print(multinomial_prob)
        multinomial_prob = multinomial.prob(3 * ones)
        tf.print(multinomial_prob)

        mixing_probs = tf.reduce_prod(mixing_probs, 0)
        mixing_probs = mixing_probs / tf.reduce_sum(mixing_probs)
        tf.print("mixing_probs prod (Cat)")
        tf.print(mixing_probs)
        dists = self.predict_experts_dists(X)
        tf.print("experts")
        tf.print(dists)
        # mixing_probs = tf.expand_dims(mixing_probs, -2)
        # mixing_probs = tf.broadcast_to(mixing_probs, dists.batch_shape)

        # tf.debugging.assert_equal(
        #     dists.batch_shape_tensor(),
        #     tf.shape(mixing_probs),
        #     message="Gating networks predict_mixing_probs(Xnew,...) and experts predict_dists(Xnew,...) dimensions do not match",
        # )
        # return tfd.MixtureSameFamily(
        #     mixture_distribution=tfd.Categorical(probs=mixing_probs),
        #     components_distribution=dists,
        # )

        # marginal_likelihood = mixture_dist.prob(Y)
        # print("marginal_likelihood")
        # print(marginal_likelihood)
        # return marginal_likelihood

    def marginal_likelihood(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Marginal likelihood (ML).

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: marginal likelihood - a Tensor with shape ()
        """
        X, Y = data
        mixture_dist = self.predict_y(X)
        marginal_likelihood = mixture_dist.prob(Y)
        return marginal_likelihood

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
