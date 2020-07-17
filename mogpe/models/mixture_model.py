from abc import ABC, abstractmethod
from typing import Optional, Tuple

import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from gpflow.models import BayesianModel, ExternalDataTrainingLossMixin
from gpflow.models.training_mixins import InputData, RegressionData

tfd = tfp.distributions


class MixtureOfExperts(BayesianModel, ABC):
    """Abstract base class for mixture of experts models.

    Given an input :math:`x` and an output :math:`y` the mixture of experts
    marginal likelihood is given by,

    .. math::
        p(y|x) = \sum_{k=1}^K \Pr(\\alpha=k | x) p(y | \\alpha=k, x)

    Assuming the mixture indicator variable :math:`\\alpha \in \{1, ...,K\}`
    the mixing probabilities are given by :math:`\Pr(\\alpha=k | x)` and are
    collectively referred to as the gating network.
    The experts are given by :math:`p(y | \\alpha=k, x)` and are responsible for
    predicting in different regions of the input space.

    Each subclass that inherits MixtureOfExperts should implement the
    maximum_log_likelihood_objective(data) method. It is used as the objective
    function to optimise the models trainable parameters.

    :param gating_network: an instance of the GatingNetworkBase class with
                            the predict_mixing_probs(Xnew) method implemented.
    :param experts: an instance of the ExpertsBase class with the
                    predict_dists(Xnew) method implemented.
    """
    def __init__(self, gating_network, experts):
        """
        :param gating_network: an instance of the GatingNetworkBase class with
                                the predict_mixing_probs(Xnew) method implemented.
        :param experts: an instance of the ExpertsBase class with the
                        predict_dists(Xnew) method implemented.
        """
        self.gating_network = gating_network
        self.experts = experts
        self.num_experts = experts.num_experts

    def predict_mixing_probs(self, Xnew: InputData, **kwargs):
        """Calculates the mixing probabilities at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param kwargs: kwargs to be passed to the gating networks
                       predict_mixing_probs() method.
        :returns: a batched Tensor with shape [..., num_test, output_dim, num_experts]
        """
        mixing_probs = self.gating_network.predict_mixing_probs(Xnew, **kwargs)
        # shape_constraints = [
        #     (mixing_probs, ["...", "num_data", "output_dim",
        #                     self.num_experts]),
        # ]
        # tf.debugging.assert_shapes(
        #     shape_constraints,
        #     message=
        #     "Mixing probabilities dimensions (from gating network) should be [..., num_data, output_dim, num_experts]"
        # )
        return self.gating_network.predict_mixing_probs(Xnew, **kwargs)

    def predict_experts_dists(self, Xnew: InputData, **kwargs) -> tf.Tensor:
        """Calculates each experts predictive distribution at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param kwargs: kwargs to be passed to the experts
                       predict_dists() method.
        :returns: a batched Tensor with shape [..., num_test, output_dim, num_experts]
        """
        dists = self.experts.predict_dists(Xnew, **kwargs)
        return dists

    def predict_y(self, Xnew: InputData, **kwargs) -> tfd.Distribution:
        # TODO should there be separate kwargs for gating and experts?
        """Predicts the mixture distribution at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param kwargs: kwargs to be passed to predict_mixing_probs and
                        predict_experts_dists
        :returns: The prediction as a TensorFlow MixtureSameFamily distribution
        """
        mixing_probs = self.predict_mixing_probs(Xnew, **kwargs)
        dists = self.predict_experts_dists(Xnew, **kwargs)
        if dists.batch_shape != tf.shape(mixing_probs):
            mixing_probs = tf.expand_dims(mixing_probs, -2)
            mixing_probs = tf.broadcast_to(mixing_probs, dists.batch_shape)

        tf.debugging.assert_equal(
            dists.batch_shape_tensor(),
            tf.shape(mixing_probs),
            message=
            "Gating networks predict_mixing_probs(Xnew,...) and experts predict_dists(Xnew,...) dimensions do not match"
        )
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixing_probs),
            components_distribution=dists)

    def predict_y_samples(self,
                          Xnew: InputData,
                          num_samples: int = 1,
                          **kwargs) -> tf.Tensor:
        """Returns samples from the predictive mixture distribution at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param num_samples: number of samples to draw
        :param kwargs: kwargs to be passed to predict_mixing_probs and
                        predict_experts_dists
        :returns: a Tensor with shape [num_samples, num_test, output_dim]
        """
        return self.predict_y(Xnew, **kwargs).sample(num_samples)


class MixtureOfSVGPExperts(MixtureOfExperts, ExternalDataTrainingLossMixin):
    """Mixture of SVGP experts using stochastic variational inference.

    Implemention of a mixture of Gaussian process (GPs) experts method where
    the gating network is also implemented using GPs.
    The model is trained with stochastic variational inference by exploiting
    the factorization achieved by sparse GPs.

    :param gating_network: an instance of the GatingNetworkBase class with
                            the predict_mixing_probs(Xnew) method implemented.
    :param experts: an instance of the ExpertsBase class with the
                    predict_dists(Xnew) method implemented.
    :param num_inducing_samples:
    :param num_data:
    """
    def __init__(self,
                 gating_network,
                 experts,
                 num_inducing_samples=1,
                 num_data=None):
        super().__init__(gating_network, experts)
        self.num_inducing_samples = num_inducing_samples
        self.num_data = num_data

    def maximum_log_likelihood_objective(
            self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Objective for maximum likelihood estimation.

        Lower bound to the log-marginal likelihood (ELBO).

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        :returns: loss - a Tensor with shape ()
        """
        with tf.name_scope('ELBO') as scope:
            X, Y = data

            # kl_gating = self.gating_network.prior_kl()
            kls_gatings = self.gating_network.prior_kls()
            kl_gating = tf.reduce_sum(kls_gatings)
            kls_experts = self.experts.prior_kls()
            kl_experts = tf.reduce_sum(kls_experts)

            with tf.name_scope('predict_mixing_probs') as scope:
                mixing_probs = self.predict_mixing_probs(
                    X, num_inducing_samples=self.num_inducing_samples)
            print("mixing_probs")
            print(mixing_probs.shape)

            with tf.name_scope('predict_experts_prob') as scope:
                batched_dists = self.predict_experts_dists(
                    X, num_inducing_samples=self.num_inducing_samples)

                Y = tf.expand_dims(Y, 0)
                Y = tf.expand_dims(Y, -1)
                expected_experts = batched_dists.prob(Y)
                print('expected experts')
                print(expected_experts.shape)

            # with tf.name_scope('predict_experts_prob') as scope:
            #     # expected_experts = self.experts.predict_prob_y(data)
            #     expected_experts = self.experts.predict_prob_y(
            #         data, {'num_inducing_samples': self.num_inducing_samples})
            #     print('expected experts')
            #     print(expected_experts.shape)
            #     # expected_experts = tf.expand_dims(expected_experts, -1)
            #     # print(expected_experts.shape)

            shape_constraints = [
                (expected_experts, [
                    "num_inducing_samples", "num_data", "output_dim",
                    "num_experts"
                ]),
                (mixing_probs, [
                    "num_inducing_samples", "num_data", "output_dim",
                    "num_experts"
                ]),
            ]
            tf.debugging.assert_shapes(
                shape_constraints,
                message="Gating network and experts dimensions do not match")
            with tf.name_scope('marginalise_indicator_variable') as scope:
                weighted_sum_over_indicator = tf.matmul(mixing_probs,
                                                        expected_experts,
                                                        transpose_b=True)
            print('marginalised indicator variable')
            print(weighted_sum_over_indicator.shape)

            # TODO correct num samples for K experts. This assumes 2 experts
            num_samples = self.num_inducing_samples**(self.num_experts + 1)
            var_exp = 1 / num_samples * tf.reduce_sum(
                tf.math.log(weighted_sum_over_indicator), axis=0)
            # tf.print(var_exp)
            print('averaged samples')
            print(var_exp.shape)
            # TODO where should output dimension be reduced?
            var_exp = tf.linalg.diag_part(var_exp)
            print('Ignore covariance in output dimension')
            print(var_exp.shape)
            print('Reduce sum to get loss')
            print(tf.reduce_sum(var_exp).shape)

            if self.num_data is not None:
                num_data = tf.cast(self.num_data, default_float())
                minibatch_size = tf.cast(tf.shape(X)[0], default_float())
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, default_float())

            return tf.reduce_sum(var_exp) * scale - kl_gating - kl_experts

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """Returns the evidence lower bound (ELBO) of the log marginal likelihood.

        :param data: data tuple (X, Y) with inputs [num_data, input_dim]
                     and outputs [num_data, ouput_dim])
        """
        return self.maximum_log_likelihood_objective(data)


def init_fake_mixture(X, Y, num_experts=2, num_inducing_samples=1):
    from mogpe.models.experts import init_fake_experts
    from mogpe.models.gating_network import init_fake_gating_network
    experts = init_fake_experts(X, Y, num_experts=2)
    gating_network = init_fake_gating_network(X, Y)
    return MixtureOfGPExperts(gating_network,
                              experts,
                              num_inducing_samples=num_inducing_samples)


if __name__ == "__main__":
    from mogpe.data.utils import load_mixture_dataset

    # Load data set
    data_file = '../../data/processed/artificial-data-used-in-paper.npz'
    data, F, prob_a_0 = load_mixture_dataset(filename=data_file,
                                             standardise=False)
    X, Y = data

    gp_mixture_model = init_fake_mixture(X,
                                         Y,
                                         num_experts=2,
                                         num_inducing_samples=4)

    # mixture_dist = gp_mixture_model.predict_y(X, {})
    # print(mixture_dist)
    # print(mixture_dist.mixture_distribution.prob(1).shape)
    # print(mixture_dist.components_distribution)

    # import matplotlib.pyplot as plt
    # plt.plot(X, mixture_dist.prob(X))
    # plt.show()

    loss = gp_mixture_model.maximum_log_likelihood_objective(data)
    print(loss.shape)
