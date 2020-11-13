from abc import ABC, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow as gpf
from gpflow import Module
from gpflow.config import default_float
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.training_mixins import InputData
from mogpe.models.gp import SVGPModel

tfd = tfp.distributions


class ExpertBase(Module, ABC):
    """Abstract base class for an individual expert.

    Each subclass that inherits ExpertBase should implement the predict_dist()
    method that returns the individual experts prediction at an input.
    """
    # @abstractmethod
    # def prior_kl(self) -> tf.Tensor:
    #     raise NotImplementedError

    @abstractmethod
    def predict_dist(self, Xnew: InputData, **kwargs):
        # def predict_dist(self, Xnew: InputData, **kwargs) -> tfd.Distribution:
        """Returns the individual experts prediction at Xnew.

        TODO: this does not return a tfd.Distribution

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: an instance of a TensorFlow Distribution
        """
        raise NotImplementedError


class ExpertsBase(Module):
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
        super().__init__(name=name)
        assert isinstance(
            experts_list,
            list), 'experts_list should be a list of ExpertBase instances'
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


class SVGPExpert(SVGPModel, ExpertBase):
    """Sparse Variational Gaussian Process Expert.

    This class inherits the prior_kl() method from the SVGPModel class
    and implements the predict_dist() method using SVGPModel's predict_y
    method.
    """
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
        """Returns the mean and (co)variance of the experts prediction at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param num_inducing_samples:
            the number of samples to draw from the inducing points joint distribution.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.
        :returns: tuple of Tensors (mean, variance),
            means shape is [num_inducing_samples, num_test, output_dim],
            if full_cov=False variance tensor has shape
            [num_inducing_samples, num_test, ouput_dim]
            and if full_cov=True,
            [num_inducing_samples, output_dim, num_test, num_test]
        """
        mu, var = self.predict_y(Xnew,
                                 num_inducing_samples=num_inducing_samples,
                                 full_cov=full_cov,
                                 full_output_cov=full_output_cov)
        return mu, var
        # return tfd.Normal(mu, tf.sqrt(var))

    # def variational_expectation(self,
    #                             data: InputData,
    #                             num_inducing_samples: int = None):
    #     X, Y = data
    #     f_mean, f_var = self.predict_f(X, num_inducing_samples, full_cov=False)
    #     # if self.num_samples is None:
    #     #     expected_prob_y = tf.exp(
    #     # self.likelihood.predict_log_density(f_mean, f_var, Y))
    #     # else:
    #     #     f_samples = expert._sample_mvn(f_mean,
    #     #                                    f_var,
    #     #                                    self.num_samples,
    #     #                                    full_cov=False)
    #     #     # f_samples = expert.predict_f_samples(X,
    #     #     #                                      num_samples_f,
    #     #     #                                      full_cov=False)
    #     #     prob_y = tf.exp(expert.likelihood._log_prob(f_samples, Y))
    #     #     expected_prob_y = 1. / self.num_samples * tf.reduce_sum(prob_y, 0)
    #     log_density = logdensities.gaussian(Y, f_mean,
    #                                         f_var + self.likelihood.variance)
    #     print(log_density.shape)
    #     expected_prob_y = tf.exp(log_density)
    #     # expected_prob_y = tf.exp(
    #     #     self.likelihood.predict_log_density(f_mean, f_var, Y))
    #     return expected_prob_y


class SVGPExperts(ExpertsBase):
    """Extension of ExpertsBase for a set of SVGPExpert experts.

    Provides an interface between a set of SVGPExpert instances and the
    MixtureOfSVGPExperts class.
    """
    def __init__(self, experts_list: List[SVGPExpert] = None, name="Experts"):
        """
        :param experts_list: a list of SVGPExpert instances with the predict_dist()
                             method implemented.
        """
        super().__init__(experts_list, name=name)
        for expert in experts_list:
            assert isinstance(expert, SVGPExpert)

    def prior_kls(self) -> tf.Tensor:
        """Returns the set of experts KL divergences as a batched tensor.

        :returns: a Tensor with shape [num_experts,]
        """
        kls = []
        for expert in self.experts_list:
            kls.append(expert.prior_kl())
        return tf.convert_to_tensor(kls)

    def predict_dists(self, Xnew: InputData, **kwargs) -> tfd.Distribution:
        """Returns the set of experts predicted dists at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: a batched tfd.Distribution with batch_shape [..., num_test, output_dim, num_experts]
        """
        mus, vars = [], []
        for expert in self.experts_list:
            mu, var = expert.predict_dist(Xnew, **kwargs)
            mus.append(mu)
            vars.append(var)
        mus = tf.stack(mus, -1)
        vars = tf.stack(vars, -1)
        return tfd.Normal(mus, tf.sqrt(vars))

    def predict_fs(self,
                   Xnew: InputData,
                   num_inducing_samples: int = None,
                   full_cov=False,
                   full_output_cov=False):
        """Returns the set experts latent function mean and (co)vars at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: a tuple of (mean, (co)var) each with shape [..., num_test, output_dim, num_experts]
        """
        mus, vars = [], []
        for expert in self.experts_list:
            mu, var = expert.predict_f(Xnew, num_inducing_samples, full_cov,
                                       full_output_cov)
            mus.append(mu)
            vars.append(var)
        mus = tf.stack(mus, -1)
        vars = tf.stack(vars, -1)
        return mus, vars


def init_fake_expert(X, Y):
    from mogpe.models.utils.model import init_inducing_variables
    output_dim = Y.shape[1]
    input_dim = X.shape[1]

    num_inducing = 7
    inducing_variable = init_inducing_variables(X, num_inducing)
    inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(inducing_variable))

    noise_var = 0.1
    lengthscale = 1.
    mean_function = gpf.mean_functions.Constant()

    likelihood = gpf.likelihoods.Gaussian(noise_var)
    # lik_list = [likelihood]
    # likelihood = gpf.likelihoods.SwitchedLikelihood(lik_list)

    kern_list = []
    for _ in range(output_dim):
        # Create multioutput kernel from kernel list
        lengthscale = tf.convert_to_tensor([lengthscale] * input_dim,
                                           dtype=default_float())
        kern_list.append(gpf.kernels.RBF(lengthscales=lengthscale))
    kernel = gpf.kernels.SeparateIndependent(kern_list)

    return SVGPExpert(kernel,
                      likelihood,
                      mean_function=mean_function,
                      inducing_variable=inducing_variable)


def init_fake_experts(X, Y, num_experts=2):
    experts_list = []
    for _ in range(num_experts):
        expert = init_fake_expert(X, Y)
        experts_list.append(expert)
    return SVGPExperts(experts_list)


if __name__ == "__main__":
    from mogpe.data.utils import load_mixture_dataset

    # Load data set
    data_file = '../../data/processed/artificial-data-used-in-paper.npz'
    data, F, prob_a_0 = load_mixture_dataset(filename=data_file,
                                             standardise=False)
    X, Y = data
    experts = init_fake_experts(X, Y)

    kls = experts.prior_kls()

    dists = experts.predict_dists(X)
