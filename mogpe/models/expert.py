import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from gpflow import Module, Parameter, logdensities
# from gpflow.conditionals import conditional, sample_conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float
from gpflow.models.training_mixins import InputData, RegressionData
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.util import inducingpoint_wrapper

from gpflow import kullback_leiblers
from gpflow.utilities import triangular, positive
from gpflow.kernels import Kernel, MultioutputKernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction, Zero

from .gp import GPModel, SVGPModel

tfd = tfp.distributions
kl = tfd.kullback_leibler


class ExpertBase(Module, ABC):
    @abstractmethod
    def prior_kl(self) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def variational_expectation(self,
                                data: InputData,
                                num_inducing_samples: int = None):
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
        print('f_mean')
        print(f_mean.shape)
        print(f_var.shape)
        print('Y')
        print(Y.shape)
        log_density = logdensities.gaussian(Y, f_mean,
                                            f_var + self.likelihood.variance)
        print(log_density.shape)
        expected_prob_y = tf.exp(log_density)
        # expected_prob_y = tf.exp(
        #     self.likelihood.predict_log_density(f_mean, f_var, Y))
        print('expected_prob_y')
        print(expected_prob_y.shape)

        # sample = True
        # num_samples = 1
        # if sample is True:
        #     f_samples = sample_mvn(f_mean,
        #                            f_var,
        #                            num_samples=num_samples,
        #                            full_cov=False)
        #     print('f_samples shape')
        #     print(f_samples.shape)
        #     prob_y = tf.exp(self.likelihood._log_prob(f_samples, Y))
        #     expected_prob_y = 1. / num_samples * tf.reduce_sum(prob_y, 0)
        # print('samples prob y shape')
        # print(expected_prob_y.shape)

        return expected_prob_y


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


if __name__ == "__main__":
    from mogpe.models.utils.data import load_mixture_dataset

    # Load data set
    data_file = '../../data/processed/artificial-data-used-in-paper.npz'
    data, F, prob_a_0 = load_mixture_dataset(filename=data_file,
                                             standardise=False)
    X, Y = data

    expert = init_fake_expert(X, Y)
    var = expert.variational_expectation(data, num_inducing_samples=10)
    print(var.shape)
