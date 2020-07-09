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
        # TODO this list is OK because it will be created before graph construction
        for expert in experts_list:
            kls.append(expert.prior_kl())
            # dists.append(expert.predict_y)
        self.prior_kls = tf.convert_to_tensor(kls)
        # self.dists = tf.convert_to_tensor(dists)

    def prior_kls(self) -> tf.Tensor:
        kl = self.prior_kls
        # tf.print('kl')
        # tf.print(kl)
        return kl

    # def prior_kls(self) -> tf.Tensor:
    #     kls = []
    #     for expert in self.experts_list:
    #         kls.append(expert.prior_kl())
    #     return kls

    # def predict_expert_y(expert, Xnew):
    #     return expert.predict_y(Xnew)

    def predict_prob_y(self, data: Tuple[tf.Tensor, tf.Tensor], kwargs={}):
        """Returns batched tensor of predicted dists"""
        # TODO this method only works for Normal dists, needs correcting
        # mus, vars = [], []
        prob_y_list = []
        X, Y = data
        for expert in self.experts_list:
            expected_prob_y = expert.variational_expectation(data, **kwargs)
            prob_y_list.append(expected_prob_y)
        prob_ys = tf.convert_to_tensor(prob_y_list)
        trailing_dims = tf.range(1, tf.rank(prob_ys))
        transpose_shape = tf.concat([trailing_dims, [0]], 0)
        return tf.transpose(prob_ys, transpose_shape)
        # return prob_y_list
        # return mus, vars

    # def predict_prob_y(self, data: Tuple[tf.Tensor, tf.Tensor], kwargs={}):
    #     """Returns batched tensor of predicted dists"""
    #     # TODO this method only works for Normal dists, needs correcting
    #     # mus, vars = [], []
    #     prob_y_list = []
    #     X, Y = data
    #     for expert in self.experts_list:
    #         f_mean, f_var = expert.predict_y(X, **kwargs)
    #         # mus.append(mu)
    #         # vars.append(var)

    #         expected_prob_y = tf.exp(
    #             expert.likelihood.predict_log_density(f_mean, f_var, Y))
    #         prob_y_list.append(expected_prob_y)
    #     prob_ys = tf.convert_to_tensor(prob_y_list)
    #     trailing_dims = tf.range(1, tf.rank(prob_ys))
    #     transpose_shape = tf.concat([trailing_dims, [0]], 0)
    #     return tf.transpose(prob_ys, transpose_shape)
    #     # return prob_y_list
    #     # return mus, vars

    def predict_dists(self, Xnew: InputData, kwargs):
        """Returns batched tensor of predicted dists"""
        # TODO this method only works for Normal dists, needs correcting
        mus, vars = [], []
        for expert in self.experts_list:
            mu, var = expert.predict_y(Xnew, **kwargs)
            mus.append(mu)
            vars.append(var)
        mus = tf.stack(mus)
        vars = tf.stack(vars)

        # move mixture dimension to last dimension
        trailing_dims_mu = tf.range(1, tf.rank(mus))
        transpose_shape_mu = tf.concat([trailing_dims_mu, [0]], 0)
        trailing_dims_var = tf.range(1, tf.rank(vars))
        transpose_shape_var = tf.concat([trailing_dims_var, [0]], 0)
        mus = tf.transpose(mus, transpose_shape_mu)
        vars = tf.transpose(vars, transpose_shape_var)
        return tfd.Normal(mus, vars)


def init_fake_experts(X, Y, num_experts=2):
    from .expert import init_fake_expert
    experts_list = []
    for _ in range(num_experts):
        expert = init_fake_expert(X, Y)
        experts_list.append(expert)
    # expert_list = [expert for _ in range(num_experts)]
    return Experts(experts_list)


if __name__ == "__main__":
    from mogpe.models.utils.data import load_mixture_dataset

    # Load data set
    data_file = '../../data/processed/artificial-data-used-in-paper.npz'
    data, F, prob_a_0 = load_mixture_dataset(filename=data_file,
                                             standardise=False)
    X, Y = data
    experts = init_fake_experts(X, Y)

    print(experts.prior_kls)

    # var_exp = experts.experts_list[0].variational_expectation(
    #     data, num_samples_inducing=10)
    # print(var_exp.shape)
    dists = experts.predict_dists(X, {})
    print(dists)
    print(dists.mean().shape)
    print(dists.variance().shape)
