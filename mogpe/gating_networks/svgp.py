#!/usr/bin/env python3
from typing import Optional, Union

import tensorflow as tf
from gpflow.kernels import Kernel
from gpflow.likelihoods import Bernoulli, Softmax
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import InputData, MeanAndVariance
from mogpe.gating_networks import GatingNetworkBase
from mogpe.gps import SVGPModel


class SVGPGatingNetwork(GatingNetworkBase, SVGPModel):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Union[Bernoulli, Softmax],
        inducing_variable,
        mean_function: MeanFunction,
        num_gating_functions: Optional[int] = 1,
        q_diag: Optional[bool] = False,
        q_mu=None,
        q_sqrt=None,
        whiten: Optional[bool] = True,
        num_data=None,
    ):
        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            num_latent_gps=num_gating_functions,
            q_diag=q_diag,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=whiten,
            num_data=num_data,
        )
        if isinstance(likelihood, Bernoulli):
            self.num_experts = 2
            assert num_gating_functions == 1
        elif isinstance(likelihood, Softmax):
            assert num_gating_functions > 1
            self.num_experts = num_gating_functions
        else:
            raise AttributeError("likelihood should be Bernoulli or Softmax")
        self.num_gating_functions = num_gating_functions

    def predict_fs(
        self,
        Xnew: InputData,
        num_inducing_samples: Optional[int] = None,
        full_cov: Optional[bool] = False,
    ) -> MeanAndVariance:
        if self.num_gating_functions == 1:
            f_mean, f_var = self.predict_f(
                Xnew, num_inducing_samples, full_cov=full_cov
            )
            if full_cov:
                F_mean = tf.concat([f_mean, -f_mean], -1)
                F_var = tf.concat([f_var, f_var], 0)
            else:
                F_mean = tf.concat([f_mean, -f_mean], -1)
                F_var = tf.concat([f_var, f_var], -1)
            return F_mean, F_var
        else:
            return self.predict_f(Xnew, num_inducing_samples, full_cov=full_cov)

    def predict_mixing_probs(
        self,
        Xnew: InputData,
        num_inducing_samples: Optional[int] = None,
        full_cov: Optional[bool] = False,
    ):
        """Compute mixing probabilities at Xnew

            Pr(\alpha = k | Xnew) = \int Pr(\alpha = k | h) p(h | Xnew) dh

        :param Xnew: test input(s) [num_data, input_dim]
        :returns: array of mixing probs for each expert [num_experts, num_data]
        """
        h_mean, h_var = self.predict_f(Xnew, num_inducing_samples, full_cov=full_cov)
        return self.predict_mixing_probs_given_h(h_mean, h_var)

    def predict_mixing_probs_given_h(
        self, h_mean: tf.Tensor, h_var: Optional[tf.Tensor] = None
    ):
        """Compute mixing probabilities given gating functions.

        if h_vars is None:
            Pr(\alpha = k | h) where h=h_mean
        else:
            \int Pr(\alpha = k | h) N(h | h_mean, h_var) dh

        :param h_mean: latent function mean or sample [num_data, num_gating_funcs]
        :param h_var: Optional latent function variance [num_data, num_gating_funcs]
        :returns: array of mixing probs for each expert [num_data, num_experts]
        """
        if h_var is not None:
            mixing_probs = self.likelihood.predict_mean_and_var(h_mean, h_var)[0]
        else:
            mixing_probs = self.likelihood.conditional_mean(h_mean)
        if self.num_gating_functions == 1:
            mixing_probs = tf.concat([mixing_probs, 1 - mixing_probs], -1)
        return mixing_probs
