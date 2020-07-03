from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Module, Parameter
from gpflow.conditionals import conditional, sample_conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData

tfd = tfp.distributions
kl = tfd.kullback_leibler


class Experts(Module):
    def __init__(self, experts_list: List = None, name="Experts"):
        """Represent the set of experts.

        This class builds tensors of prior KL divergence and """
        assert isinstance(
            experts_list,
            list), 'experts_list should be a list of Expert instances'
        self.num_experts = len(experts_list)
        self.experts_list = experts_list
        kls, dists = [], []
        for expert in experts_list:
            kls.append(expert.prior_kl())
            # dists.append(expert.predict_y)
        self.prior_kls = tf.convert_to_tensor(kls)
        # self.dists = tf.convert_to_tensor(dists)

    def prior_kls(self) -> tf.Tensor:
        return self.prior_kls

    def predict_dists(self, Xnew):
        mus, vars = [], []
        for expert in self.experts_list:
            mu, var = expert.predict_y(Xnew)
            mus.append(mu)
            vars.append(var)
        return tfd.Normal(mus, vars)


def init_fake_experts(X, Y, num_experts=2):
    from expert import init_fake_expert
    expert = init_fake_expert(X, Y)
    expert_list = [expert for _ in range(num_experts)]
    return Experts(expert_list)


if __name__ == "__main__":
    from src.models.utils.data import load_mixture_dataset

    # Load data set
    data_file = '../../data/processed/artificial-data-used-in-paper.npz'
    data, F, prob_a_0 = load_mixture_dataset(filename=data_file,
                                             standardise=False)
    X, Y = data
    experts = init_fake_experts(X, Y)

    print(experts.prior_kls)

    var_exp = experts.experts_list[0].variational_expectation(
        data, num_samples_inducing=10)
    print(var_exp.shape)
    dists = experts.predict_dists(X)
    print(dists)
    print(dists.mean().shape)
    print(dists.variance().shape)
