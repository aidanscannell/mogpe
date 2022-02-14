#!/usr/bin/env python3
from abc import ABC, abstractmethod

import tensorflow as tf
from gpflow import Module
from gpflow.models.model import InputData, MeanAndVariance


class GatingNetworkBase(Module, ABC):
    """Abstract base class for the gating network."""

    @abstractmethod
    def predict_fs(self, Xnew: InputData, **kwargs) -> MeanAndVariance:
        """Calculates the set of gating function posteriors at Xnew

        :param Xnew: inputs with shape [num_test, input_dim]
        TODO correct dimensions
        :returns: mean and var batched Tensors with shape [..., num_test, 1, num_experts]
        """
        raise NotImplementedError

    @abstractmethod
    def predict_mixing_probs(self, Xnew: InputData, **kwargs) -> tf.Tensor:
        """Calculates the set of experts mixing probabilities at Xnew :math:`\{\Pr(\\alpha=k | x)\}^K_{k=1}`

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: a batched Tensor with shape [..., num_test, 1, num_experts]
        """
        raise NotImplementedError

    def call(self, Xnew: InputData, **kwargs) -> tf.Tensor:
        return self.predict_mixing_probs(Xnew, kwargs)
