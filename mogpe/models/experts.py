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

import numpy as np

from gpflow import Module, Parameter, logdensities
from gpflow.models.util import inducingpoint_wrapper

from gpflow.utilities import triangular, positive
from gpflow.kernels import Kernel, MultioutputKernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction, Zero

from .gp import GPModel, SVGPModel

tfd = tfp.distributions


class ExpertsBase(Module):
    def __init__(self, experts_list: List = None, name="Experts"):
        """Provides an interface between ExpertBase and MixtureOfExperts.

        It should return the KL divergence for each experts inducing points
        and the set of experts predictions at an input (as a batched TensorFlow
        distribution)

        :param experts_list: A list of experts that inherit from ExpertBase
        """
        super().__init__(name=name)
        assert isinstance(
            experts_list,
            list), 'experts_list should be a list of ExpertBase instances'
        self.num_experts = len(experts_list)
        self.experts_list = experts_list

    def prior_kls(self) -> tf.Tensor:
        raise NotImplementedError

    def predict_dists(self, Xnew: InputData, kwargs) -> tfd.Distribution:
        """Returns batched tensor of predicted dists"""
        raise NotImplementedError


class SVGPExperts(ExpertsBase):
    def __init__(self, experts_list: List = None, name="Experts"):
        """Implementation of ExpertsBase for Normally distributed experts"""
        super().__init__(experts_list, name=name)

    def prior_kls(self) -> tf.Tensor:
        kls = []
        for expert in self.experts_list:
            kls.append(expert.prior_kl())
        return tf.convert_to_tensor(kls)

    def predict_dists(self, Xnew: InputData, kwargs) -> tfd.Distribution:
        """Returns a batched TensorFlow distribution at Xnew with
        shape [num_inducing_samples, num_data, ouput_dim, num_experts]"""
        # TODO this method only works for Normal dists, needs correcting
        mus, vars = [], []
        for expert in self.experts_list:
            mu, var = expert.predict_dist(Xnew, **kwargs)
            mus.append(mu)
            vars.append(var)
        mus = tf.stack(mus, -1)
        vars = tf.stack(vars, -1)
        return tfd.Normal(mus, tf.sqrt(vars))


class ExpertBase(Module, ABC):
    @abstractmethod
    def prior_kl(self) -> tf.Tensor:
        raise NotImplementedError

    # @abstractmethod
    # def variational_expectation(self,
    #                             data: InputData,
    #                             num_inducing_samples: int = None):
    #     raise NotImplementedError
    @abstractmethod
    def predict_dist(self, data: InputData, num_inducing_samples: int = None):
        raise NotImplementedError


class SVGPExpert(SVGPModel, ExpertBase):
    def __init__(self,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 inducing_variable,
                 mean_function: MeanFunction = None,
                 num_latent_gps: int = 1,
                 q_diag: bool = False,
                 q_mu=None,
                 q_sqrt=None,
                 whiten: bool = True,
                 num_data=None):
        super().__init__(kernel, likelihood, inducing_variable, mean_function,
                         num_latent_gps, q_diag, q_mu, q_sqrt, whiten,
                         num_data)

    def predict_dist(self,
                     Xnew: InputData,
                     num_inducing_samples: int = None,
                     full_cov: bool = False,
                     full_output_cov: bool = False):
        mu, var = self.predict_y(Xnew,
                                 num_inducing_samples=num_inducing_samples,
                                 full_cov=full_cov,
                                 full_output_cov=full_output_cov)
        return mu, var
        # return tfd.Normal(mu, tf.sqrt(var))

    def variational_expectation(self,
                                data: InputData,
                                num_inducing_samples: int = None):
        X, Y = data
        f_mean, f_var = self.predict_f(X, num_inducing_samples, full_cov=False)
        # if self.num_samples is None:
        #     expected_prob_y = tf.exp(
        # self.likelihood.predict_log_density(f_mean, f_var, Y))
        # else:
        #     f_samples = expert._sample_mvn(f_mean,
        #                                    f_var,
        #                                    self.num_samples,
        #                                    full_cov=False)
        #     # f_samples = expert.predict_f_samples(X,
        #     #                                      num_samples_f,
        #     #                                      full_cov=False)
        #     prob_y = tf.exp(expert.likelihood._log_prob(f_samples, Y))
        #     expected_prob_y = 1. / self.num_samples * tf.reduce_sum(prob_y, 0)
        log_density = logdensities.gaussian(Y, f_mean,
                                            f_var + self.likelihood.variance)
        print(log_density.shape)
        expected_prob_y = tf.exp(log_density)
        # expected_prob_y = tf.exp(
        #     self.likelihood.predict_log_density(f_mean, f_var, Y))
        return expected_prob_y


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
