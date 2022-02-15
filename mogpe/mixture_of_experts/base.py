#!/usr/bin/env python3
from abc import ABC

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import BayesianModel
from gpflow.models.training_mixins import InputData
from mogpe.experts import ExpertsBase
from mogpe.gating_networks import GatingNetworkBase

tfd = tfp.distributions


class MixtureOfExperts(BayesianModel, tf.keras.Model, ABC):
    """Abstract base class for mixture of experts models.

    Given an input :math:`x` and an output :math:`y` the mixture of experts
    marginal likelihood is given by,

    .. math::
        p(y|x) = \sum_{k=1}^K \Pr(\\alpha=k | x) p(y | \\alpha=k, x)

    Assuming the expert indicator variable :math:`\\alpha \in \{1, ...,K\}`
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

    def __init__(self, gating_network: GatingNetworkBase, experts: ExpertsBase):
        """
        :param gating_network: an instance of the GatingNetworkBase class with
                                the predict_mixing_probs(Xnew) method implemented.
        :param experts: an instance of the ExpertsBase class with the
                        predict_dists(Xnew) method implemented.
        """
        super().__init__()
        # tf.keras.Model.__init__(self)
        # BayesianModel.__init__(self)
        assert isinstance(gating_network, GatingNetworkBase)
        self.gating_network = gating_network
        assert isinstance(experts, ExpertsBase)
        self.experts = experts
        self.num_experts = experts.num_experts

    # def call(self, Xnew: InputData):
    #     return tf.reduce_sum(Xnew)

    def predict_mixing_probs(self, Xnew: InputData, **kwargs):
        """Calculates the mixing probabilities at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param kwargs: kwargs to be passed to the gating networks
                       predict_mixing_probs() method.
        :returns: a batched Tensor with shape [..., num_test, num_experts]
        """
        with tf.name_scope("predict_mixing_probs") as scope:
            mixing_probs = self.gating_network.predict_mixing_probs(Xnew, **kwargs)
        num_data = Xnew.shape[0]
        shape_constraints = [
            (mixing_probs, [num_data, self.num_experts]),
        ]
        tf.debugging.assert_shapes(
            shape_constraints,
            message="Mixing probabilities dimensions (from gating network) should be [num_data, num_experts]",
        )
        return self.gating_network.predict_mixing_probs(Xnew, **kwargs)

    def predict_experts_dists(self, Xnew: InputData, **kwargs) -> tf.Tensor:
        """Calculates each experts predictive distribution at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param kwargs: kwargs to be passed to the experts
                       predict_dists() method.
        :returns: a batched Tensor with shape [..., num_test, output_dim, num_experts]
        """
        with tf.name_scope("predict_experts_dists") as scope:
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
        # if dists.batch_shape != tf.shape(mixing_probs):
        #     # mixing_probs = tf.expand_dims(mixing_probs, -2)
        #     mixing_probs = tf.broadcast_to(mixing_probs, dists.batch_shape)
        mixing_probs = tf.expand_dims(mixing_probs, -2)
        mixing_probs = tf.broadcast_to(mixing_probs, dists.batch_shape)

        tf.debugging.assert_equal(
            dists.batch_shape_tensor(),
            tf.shape(mixing_probs),
            message="Gating networks predict_mixing_probs(Xnew,...) and experts predict_dists(Xnew,...) dimensions do not match",
        )
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixing_probs),
            components_distribution=dists,
        )

    def predict_y_samples(
        self, Xnew: InputData, num_samples: int = 1, **kwargs
    ) -> tf.Tensor:
        """Returns samples from the predictive mixture distribution at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param num_samples: number of samples to draw
        :param kwargs: kwargs to be passed to predict_mixing_probs and
                        predict_experts_dists
        :returns: a Tensor with shape [num_samples, num_test, output_dim]
        """
        return self.predict_y(Xnew, **kwargs).sample(num_samples)

    def get_config(self):
        config = super.get_config()
        config.update(
            {
                "experts": self.experts.getconfig(),
                "gating_network": self.gating_network.get_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # def get_config(self):
    #     """Returns the config for keras"""
    #     config = {
    #         "predict_y_samples_fn": _serialize_function(self.predict_y_samples),
    #         "predict_y_fn": _serialize_function(self.predict_y),
    #         "predict_mixing_probs_fn": _serialize_function(self.predict_mixing_probs),
    #         "predict_experts_dists_fn": _serialize_function(self.predict_experts_dists),
    #     }
    #     return config
    # base_config = super(DistributionLambda, self).get_config()
    # return dict(list(base_config.items()) + list(config.items()))
