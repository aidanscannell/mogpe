"""
This code is copied (and lightly modified) from `GPflow <https://github.com/GPflow/GPflow>`_.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Module, Parameter, kullback_leiblers
from gpflow.conditionals import conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float
from gpflow.kernels import Kernel, MultioutputKernel
from gpflow.likelihoods import Likelihood, SwitchedLikelihood
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData
from gpflow.models.util import inducingpoint_wrapper
from gpflow.utilities import positive, triangular

from .conditionals import separate_independent_conditional

tfd = tfp.distributions
kl = tfd.kullback_leibler


class GPModel(Module, ABC):
    def __init__(self,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: Optional[MeanFunction] = None,
                 num_latent_gps: int = None):
        super().__init__(name='Expert')
        assert num_latent_gps is not None, "GPModel requires specification of num_latent_gps"
        self.num_latent_gps = num_latent_gps
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        self.kernel = kernel
        self.likelihood = likelihood

    @staticmethod
    def calc_num_latent_gps_from_data(data, kernel: Kernel,
                                      likelihood: Likelihood) -> int:
        """
        Calculates the number of latent GPs required based on the data as well
        as the type of kernel and likelihood.
        """
        _, Y = data
        output_dim = Y.shape[-1]
        return GPModel.calc_num_latent_gps(kernel, likelihood, output_dim)

    @staticmethod
    def calc_num_latent_gps(kernel: Kernel, likelihood: Likelihood,
                            output_dim: int) -> int:
        """
        Calculates the number of latent GPs required given the number of
        outputs `output_dim` and the type of likelihood and kernel.
        Note: It's not nice for `GPModel` to need to be aware of specific
        likelihoods as here. However, `num_latent_gps` is a bit more broken in
        general, we should fix this in the future. There are also some slightly
        problematic assumptions re the output dimensions of mean_function.
        See https://github.com/GPflow/GPflow/issues/1343
        """
        if isinstance(kernel, MultioutputKernel):
            # MultioutputKernels already have num_latent_gps attributes
            num_latent_gps = kernel.num_latent_gps
        elif isinstance(likelihood, SwitchedLikelihood):
            # the SwitchedLikelihood partitions/stitches based on the last
            # column in Y, but we should not add a separate latent GP for this!
            # hence decrement by 1
            num_latent_gps = output_dim - 1
            assert num_latent_gps > 0
        else:
            num_latent_gps = output_dim

        return num_latent_gps

    @abstractmethod
    def predict_f(self,
                  Xnew: InputData,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        raise NotImplementedError

    def predict_f_samples(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = 1,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.

        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, D]
            where N is the number of rows and D is the input dimension of each point.
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples and P is the number of outputs.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.
            Currently, the method does not support `full_output_cov=True` and `full_cov=True`.
        """
        if full_cov and full_output_cov:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not supported."
            )

        # check below for shape info
        mean, cov = self.predict_f(Xnew,
                                   full_cov=full_cov,
                                   full_output_cov=full_output_cov)
        if full_cov:
            # mean: [..., N, P]
            # cov: [..., P, N, N]
            mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
            samples = sample_mvn(mean_for_sample,
                                 cov,
                                 full_cov,
                                 num_samples=num_samples)  # [..., (S), P, N]
            samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        else:
            # mean: [..., N, P]
            # cov: [..., N, P] or [..., N, P, P]
            samples = sample_mvn(mean,
                                 cov,
                                 full_output_cov,
                                 num_samples=num_samples)  # [..., (S), N, P]
        return samples  # [..., (S), N, P]

    def predict_y(self,
                  Xnew: InputData,
                  num_inducing_samples: int = None,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        """Compute mean and variance at Xnew."""
        raise NotImplementedError


class SVGPModel(GPModel):
    def __init__(
            self,
            kernel: Kernel,
            likelihood: Likelihood,
            inducing_variable,
            mean_function: MeanFunction = None,
            num_latent_gps: int = 1,
            # num_inducing_samples: Optional[int] = None,
            q_diag: bool = False,
            q_mu=None,
            q_sqrt=None,
            whiten: bool = True,
            num_data=None):
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        self.num_inducing = len(self.inducing_variable)
        self._init_variational_parameters(self.num_inducing, q_mu, q_sqrt,
                                          q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
            Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
            If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
            and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
            initializes them, their shape depends on `num_inducing` and `q_diag`.
            Note: most often the comments refer to the number of observations (=output dimensions) with P,
            number of latent GPs with L, and number of inducing points M. Typically P equals L,
            but when certain multioutput kernels are used, this can change.
            Parameters
            ----------
            :param num_inducing: int
                Number of inducing variables, typically refered to as M.
            :param q_mu: np.array or None
                Mean of the variational Gaussian posterior. If None the function will initialise
                the mean with zeros. If not None, the shape of `q_mu` is checked.
            :param q_sqrt: np.array or None
                Cholesky of the covariance of the variational Gaussian posterior.
                If None the function will initialise `q_sqrt` with identity matrix.
                If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
            :param q_diag: bool
                Used to check if `q_mu` and `q_sqrt` have the correct shape or to
                construct them with the correct shape. If `q_diag` is true,
                `q_sqrt` is two dimensional and only holds the square root of the
                covariance diagonal elements. If False, `q_sqrt` is three dimensional.
            """
        q_mu = np.zeros(
            (num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent_gps),
                               dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [
                    np.eye(num_inducing, dtype=default_float())
                    for _ in range(self.num_latent_gps)
                ]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt,
                                        transform=triangular())  # [P, M, M]
        else:
            if q_diag is True:
                assert q_sqrt.ndim == 2
                self.num_latent_gps = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt,
                                        transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent_gps = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt,
                                        transform=triangular())  # [L|P, M, M]

    def prior_kl(self) -> tf.Tensor:
        with tf.name_scope('KL_divergence') as scope:
            return kullback_leiblers.prior_kl(self.inducing_variable,
                                              self.kernel,
                                              self.q_mu,
                                              self.q_sqrt,
                                              whiten=self.whiten)

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
