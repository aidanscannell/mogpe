#!/usr/bin/env python3
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow as gpf
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.training_mixins import InputData

from mogpe.gps import SVGPModel
from mogpe.experts import ExpertBase, ExpertsBase

tfd = tfp.distributions


class SVGPExpert(SVGPModel, ExpertBase):
    """Sparse Variational Gaussian Process Expert.

    This class inherits the prior_kl() method from the SVGPModel class
    and implements the predict_dist() method using SVGPModel's predict_y
    method.
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
        num_data=None,
    ):
        super().__init__(
            kernel,
            likelihood,
            inducing_variable,
            mean_function,
            num_latent_gps,
            q_diag,
            q_mu,
            q_sqrt,
            whiten,
            num_data,
        )

    def predict_dist(
        self,
        Xnew: InputData,
        num_inducing_samples: int = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
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
        mu, var = self.predict_y(
            Xnew,
            num_inducing_samples=num_inducing_samples,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
        )
        return mu, var


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

    def predict_fs(
        self,
        Xnew: InputData,
        num_inducing_samples: int = None,
        full_cov=False,
        full_output_cov=False,
    ) -> MeanAndVariance:
        """Returns the set experts latent function mean and (co)vars at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: a tuple of (mean, (co)var) each with shape [..., num_test, output_dim, num_experts]
        """
        mus, vars = [], []
        for expert in self.experts_list:
            mu, var = expert.predict_f(
                Xnew, num_inducing_samples, full_cov, full_output_cov
            )
            mus.append(mu)
            vars.append(var)
        mus = tf.stack(mus, -1)
        vars = tf.stack(vars, -1)
        return mus, vars

    def predict_ys(
        self,
        Xnew: InputData,
        num_inducing_samples: int = None,
        full_cov=False,
        full_output_cov=False,
    ) -> MeanAndVariance:
        """Returns the set of experts predictions mean and (co)vars at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: a tuple of (mean, (co)var) each with shape [..., num_test, output_dim, num_experts]
        """
        dists = self.predict_dists(
            Xnew,
            num_inducing_samples=num_inducing_samples,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
        )
        return dists.mean(), dists.variance()

    def likelihoods(self) -> List[Likelihood]:
        likelihoods = []
        for expert in self.experts_list:
            likelihoods.append(expert.likelihood)
        return likelihoods

    def noise_variances(self) -> List[tf.Tensor]:
        variances = []
        for expert in self.experts_list:
            variances.append(expert.likelihood.variance)
        return variances
