#!/usr/bin/env python3
import abc
from typing import List, Optional

import tensorflow as tf
import tensorflow_probability as tfp
from mogpe.custom_types import InputData, MixingProb
from gpflow.models import BayesianModel

from .. import EXPERT_OBJECTS, GATING_NETWORK_OBJECTS
from ..experts import ExpertBase
from ..gating_networks import GatingNetworkBase

tfd = tfp.distributions
tfpl = tfp.layers


# class MixtureOfExpertsBase(gpf.models.BayesianModel, abc.ABC):
# class MixtureOfExpertsBase(tf.keras.Model, abc.ABC):
class MixtureOfExpertsBase(tf.keras.Model, BayesianModel, abc.ABC):
    r"""Interface for mixture of experts models.

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
    """

    def __init__(
        self,
        experts_list: List[ExpertBase],
        gating_network: GatingNetworkBase,
        name: str = "MixtureOfExperts",
    ):
        super().__init__(name=name)
        self._experts_list = experts_list
        self._gating_network = gating_network

        for expert in experts_list:
            assert isinstance(expert, ExpertBase)
        assert isinstance(gating_network, GatingNetworkBase)
        assert gating_network.num_experts == self.num_experts

        # self.mixture_layer = tfpl.MixtureSameFamily(self.num_experts, tfpl.IndependentNormal(self.)),

    def call(self, Xnew: InputData, training: Optional[bool] = True):
        if training:
            return self.predict_y(Xnew)
        else:
            raise NotImplementedError(
                "mixture of experts model should implement functionality for training=True"
            )

    def predict_y(self, Xnew: InputData, **kwargs) -> tfd.Mixture:
        # TODO should there be separate kwargs for gating and experts?
        """Predicts the mixture distribution at Xnew.

        Mixture dist has
        batch_shape == [NumData]
        event_shape == [OutputDim] # TODO is this true for multi output setting and full_cov?
        """
        expert_indicator_categorical_dist = self.gating_network(Xnew, **kwargs)
        experts_dists = self.predict_experts_dists(Xnew, **kwargs)
        return tfd.Mixture(
            cat=expert_indicator_categorical_dist, components=experts_dists
        )

    def predict_experts_dists(
        self, Xnew: InputData, **kwargs
    ) -> List[tfd.Distribution]:
        """Calculates each experts predictive distribution at Xnew.

        Each expert's dist has shape [NumData, OutputDim, NumExperts]
        if not full_cov:
            batch_shape == [[NumData], [NumData], ...] with len(batch_shape)=NumExperts
            event_shape == [[OutputDim], [OutputDim], ...] with len(event_shape)=NumExperts
        """
        return [expert(Xnew, **kwargs) for expert in self.experts_list]

    # def predict_mixing_probs(self, Xnew: InputData, **kwargs) -> MixingProb:
    #     """Calculates the mixing probabilities at Xnew."""
    #     mixing_probs = self.gating_network.predict_mixing_probs(Xnew, **kwargs)
    #     num_data = Xnew.shape[0]
    #     shape_constraints = [
    #         (mixing_probs, [num_data, self.num_experts]),
    #     ]
    #     tf.debugging.assert_shapes(
    #         shape_constraints,
    #         message="Mixing probabilities dimensions (from gating network) should be [num_data, num_experts]",
    #     )
    #     return mixing_probs

    @property
    def experts_list(self) -> List[ExpertBase]:
        return self._experts_list

    @property
    def gating_network(self) -> GatingNetworkBase:
        return self._gating_network

    @property
    def num_experts(self) -> int:
        return len(self.experts_list)

    # def predict_y_samples(
    #     self, Xnew: InputData, num_samples: int = 1, **kwargs
    # ) -> ttf.Tensor3[NumSamples, NumData, OutputDim]:
    #     """Returns samples from the predictive mixture distribution at Xnew."""
    #     return self.predict_y(Xnew, **kwargs).sample(num_samples)

    def get_config(self):
        experts_list = []
        for expert in self.experts_list:
            experts_list.append(tf.keras.layers.serialize(expert))
        return {
            "experts_list": experts_list,
            "gating_network": tf.keras.layers.serialize(self.gating_network),
        }

    @classmethod
    def from_config(cls, cfg: dict):
        expert_list = []
        for expert_cfg in cfg["experts_list"]:
            expert_list.append(
                tf.keras.layers.deserialize(expert_cfg, custom_objects=EXPERT_OBJECTS)
            )
        gating_network = tf.keras.layers.deserialize(
            cfg["gating_network"], custom_objects=GATING_NETWORK_OBJECTS
        )
        return cls(experts_list=expert_list, gating_network=gating_network)
