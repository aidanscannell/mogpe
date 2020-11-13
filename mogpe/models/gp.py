from abc import ABC, abstractmethod
from typing import Optional, Tuple

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import kullback_leiblers
from gpflow.conditionals import conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models import SVGP
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData

tfd = tfp.distributions
kl = tfd.kullback_leibler


class SVGPModel(SVGP):
    """Extension of GPflow's SVGP class with inducing point sampling.

    It reimplements predict_f and predict_y with an argument to set the number of samples
    (num_inducing_samples) to draw form the inducing outputs distribution.
    If num_inducing_samples is None then the standard functionality is achieved, i.e. the inducing points
    are analytically marginalised.
    If num_inducing_samples=3 then 3 samples are drawn from the inducing ouput distribution and the standard
    GP conditional is called (q_sqrt=None). The results for each sample are stacked on the leading dimension
    and the user now has the ability marginalise them outside of this class.
    """
    def __init__(
            self,
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
        super().__init__(kernel, likelihood, inducing_variable, mean_function=mean_function, num_latent_gps=num_latent_gps,
                         q_diag=q_diag, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten, num_data=num_data)

    def sample_inducing_points(self, num_samples: int = None) -> tf.Tensor:
        """Returns samples from the inducing point distribution.

        The distribution is given by,

        .. math::
            q \sim \mathcal{N}(q\_mu, q\_sqrt q\_sqrt^T)

        :param num_samples: the number of samples to draw
        :returns: samples with shape [num_samples, num_data, output_dim]
        """
        mu = tf.transpose(self.q_mu, [1, 0])
        q_dist = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=self.q_sqrt,
            validate_args=False,
            allow_nan_stats=True,
            name='MultivariateNormalQ')
        inducing_samples = q_dist.sample(num_samples)
        return tf.transpose(inducing_samples, [0, 2, 1])

    def predict_f(self,
                  Xnew: InputData,
                  num_inducing_samples: int = None,
                  full_cov=False,
                  full_output_cov=False) -> MeanAndVariance:
        """"Compute mean and (co)variance of latent function at Xnew.

        If num_inducing_samples is not None then sample inducing points instead
        of analytically integrating them. This is required in the mixture of
        experts lower bound."""
        with tf.name_scope('predict_f') as scope:
            if num_inducing_samples is None:
                q_mu = self.q_mu
                q_sqrt = self.q_sqrt
                mu, var = conditional(Xnew,
                                      self.inducing_variable,
                                      self.kernel,
                                      q_mu,
                                      q_sqrt=q_sqrt,
                                      full_cov=full_cov,
                                      white=self.whiten,
                                      full_output_cov=full_output_cov)
            else:
                q_mu = self.sample_inducing_points(num_inducing_samples)
                q_sqrt = None

                @tf.function
                def single_sample_conditional(q_mu):
                    # TODO requires my hack/fix to gpflow's separate_independent_conditional
                    return conditional(Xnew,
                                       self.inducing_variable,
                                       self.kernel,
                                       q_mu,
                                       q_sqrt=q_sqrt,
                                       full_cov=full_cov,
                                       white=self.whiten,
                                       full_output_cov=full_output_cov)

                mu, var = tf.map_fn(single_sample_conditional,
                                    q_mu,
                                    dtype=(default_float(), default_float()))
            return mu + self.mean_function(Xnew), var


    def predict_y(self,
                  Xnew: InputData,
                  num_inducing_samples: int = None,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        """Compute mean and variance at Xnew."""
        f_mean, f_var = self.predict_f(
            Xnew,
            num_inducing_samples=num_inducing_samples,
            full_cov=full_cov,
            full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)


def init_fake_svgp(X, Y):
    from mogpe.models.utils.model import init_inducing_variables
    output_dim = Y.shape[1]
    input_dim = X.shape[1]

    num_inducing = 30
    inducing_variable = init_inducing_variables(X, num_inducing)

    inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(inducing_variable))

    noise_var = 0.1
    lengthscale = 1.
    mean_function = gpf.mean_functions.Constant()
    likelihood = gpf.likelihoods.Gaussian(noise_var)

    kern_list = []
    for _ in range(output_dim):
        # Create multioutput kernel from kernel list
        lengthscale = tf.convert_to_tensor([lengthscale] * input_dim,
                                           dtype=default_float())
        kern_list.append(gpf.kernels.RBF(lengthscales=lengthscale))
    kernel = gpf.kernels.SeparateIndependent(kern_list)

    return SVGPModel(kernel,
                     likelihood,
                     mean_function=mean_function,
                     inducing_variable=inducing_variable)


if __name__ == "__main__":
    # Load data set
    from mogpe.models.utils.data import load_mixture_dataset
    data_file = '../../data/processed/artificial-data-used-in-paper.npz'
    data, F, prob_a_0 = load_mixture_dataset(filename=data_file,
                                             standardise=False)
    X, Y = data

    svgp = init_fake_svgp(X, Y)

    # samples = svgp.predict_f_samples(X, 3)
    # mu, var = svgp.predict_y(X)
    # mu, var = svgp.predict_f(X, 10, full_cov=True)
    mu, var = svgp.predict_f(X, 10, full_cov=True)
    print(mu.shape)
    print(var.shape)
    # samples = svgp.sample_inducing_points(3)
    # print(samples.shape)
