#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import List

import tensorflow_probability as tfp

import tensorflow as tf
from gpflow import Module
from gpflow.models.model import InputData

tfd = tfp.distributions


# class ExpertBase(Module, ABC):
# class ExpertBase(tf.keras.Model):
class ExpertBase(tf.keras.Model, Module):
    """Abstract base class for an individual expert.

    Each subclass that inherits ExpertBase should implement the predict_dist()
    method that returns the individual experts prediction at an input.
    """

    @abstractmethod
    def predict_dist(self, Xnew: InputData, **kwargs):
        # def predict_dist(self, Xnew: InputData, **kwargs) -> tfd.Distribution:
        """Returns the individual experts prediction at Xnew.

        TODO: this does not return a tfd.Distribution

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: an instance of a TensorFlow Distribution
        """
        raise NotImplementedError


# class ExpertsBase(Module, ABC):
# class ExpertsBase(tf.keras.Model):
class ExpertsBase(tf.keras.Model, Module):
    """Abstract base class for a set of experts.

    Provides an interface between ExpertBase and MixtureOfExperts.
    Each subclass that inherits ExpertsBase should implement the predict_dists()
    method that returns the set of experts predictions at an input (as a
    batched TensorFlow distribution).
    """

    def __init__(self, experts_list: List[ExpertBase] = None, name="Experts"):
        """
        :param experts_list: A list of experts that inherit from ExpertBase
        """
        # super().__init__(name=name)
        super().__init__()
        assert isinstance(
            experts_list, list
        ), "experts_list should be a list of ExpertBase instances"
        for expert in experts_list:
            assert isinstance(expert, ExpertBase)
        self.num_experts = len(experts_list)
        self.experts_list = experts_list

    @abstractmethod
    def predict_dists(self, Xnew: InputData, **kwargs) -> tfd.Distribution:
        """Returns the set of experts predicted dists at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: a batched tfd.Distribution with batch_shape [..., num_test, output_dim, num_experts]
        """
        raise NotImplementedError
