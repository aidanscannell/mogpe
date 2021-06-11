#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import List, Union, Optional

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Module
from gpflow.conditionals import conditional, sample_conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float
from gpflow.likelihoods import Bernoulli, Likelihood, Softmax
from gpflow.mean_functions import MeanFunction
from gpflow.kernels import Kernel
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.util import inducingpoint_wrapper
from gpflow.quadrature import ndiag_mc

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
        # name="SVGPGatingNetwork",
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
        # self.name = name
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


# class SVGPGatingFunction(SVGPModel):
#     # TODO either remove likelihood or use Bernoulli/Softmax
#     def __init__(
#         self,
#         kernel,
#         inducing_variable,
#         mean_function,
#         num_latent_gps=1,
#         q_diag=False,
#         q_mu=None,
#         q_sqrt=None,
#         whiten=True,
#         num_data=None,
#     ):
#         super().__init__(
#             kernel,
#             likelihood=None,
#             inducing_variable=inducing_variable,
#             mean_function=mean_function,
#             num_latent_gps=num_latent_gps,
#             q_diag=q_diag,
#             q_mu=q_mu,
#             q_sqrt=q_sqrt,
#             whiten=whiten,
#             num_data=num_data,
#         )


# class SVGPGatingNetworkBase(GatingNetworkBase):
#     """Abstract base class for gating networks based on SVGPs."""

#     def __init__(
#         self,
#         gating_function_list: List[SVGPGatingFunction] = None,
#         name="GatingNetwork",
#     ):
#         super().__init__(name=name)
#         assert isinstance(gating_function_list, List)
#         for gating_function in gating_function_list:
#             assert isinstance(gating_function, SVGPGatingFunction)
#         self.gating_function_list = gating_function_list
#         self.num_experts = len(gating_function_list)

#     def prior_kls(self) -> tf.Tensor:
#         """Returns the set of experts KL divergences as a batched tensor.

#         :returns: a Tensor with shape [num_experts,]
#         """
#         kls = []
#         for gating_function in self.gating_function_list:
#             kls.append(gating_function.prior_kl())
#         return tf.convert_to_tensor(kls)

#     def predict_fs(
#         self, Xnew: InputData, num_inducing_samples: int = None
#     ) -> MeanAndVariance:
#         Fmu, Fvar = [], []
#         for gating_function in self.gating_function_list:
#             f_mu, f_var = gating_function.predict_f(Xnew, num_inducing_samples)
#             Fmu.append(f_mu)
#             Fvar.append(f_var)
#         # Fmu = tf.stack(Fmu)
#         # Fvar = tf.stack(Fvar)
#         Fmu = tf.stack(Fmu, -1)
#         Fvar = tf.stack(Fvar, -1)
#         return Fmu, Fvar


# class SVGPGatingNetworkMulti(SVGPGatingNetworkBase):
#     # TODO either remove likelihood or use Bernoulli/Softmax
#     def __init__(
#         self,
#         gating_function_list: List[SVGPGatingFunction] = None,
#         likelihood: Likelihood = None,
#         name="GatingNetwork",
#     ):
#         super().__init__(gating_function_list, name=name)
#         # assert isinstance(gating_function_list, List)
#         # for gating_function in gating_function_list:
#         #     assert isinstance(gating_function, SVGPGatingFunction)
#         # self.gating_function_list = gating_function_list
#         # self.num_experts = len(gating_function_list)

#         if likelihood is None:
#             self.likelihood = Softmax(num_classes=self.num_experts)
#             # self.likelihood = Softmax(num_classes=1)
#         else:
#             self.likelihood = likelihood

#     def predict_mixing_probs(
#         self, Xnew: InputData, num_inducing_samples: int = None
#     ) -> tf.Tensor:

#         mixing_probs = []
#         Fmu, Fvar = [], []
#         for gating_function in self.gating_function_list:
#             # num_inducing_samples = None
#             f_mu, f_var = gating_function.predict_f(Xnew, num_inducing_samples)
#             Fmu.append(f_mu)
#             Fvar.append(f_var)
#         # Fmu = tf.stack(Fmu)
#         # Fvar = tf.stack(Fvar)
#         Fmu = tf.stack(Fmu, -1)
#         Fvar = tf.stack(Fvar, -1)
#         # Fmu = tf.concat(Fmu, -1)
#         # Fvar = tf.concat(Fvar, -1)
#         if num_inducing_samples is None:
#             Fmu = tf.transpose(Fmu, [1, 0, 2])
#             Fvar = tf.transpose(Fvar, [1, 0, 2])
#         else:
#             Fmu = tf.transpose(Fmu, [2, 0, 1, 3])
#             Fvar = tf.transpose(Fvar, [2, 0, 1, 3])

#         # TODO output dimension is always 1 so delete this
#         def single_output_predict_mean(args):
#             Fmu, Fvar = args

#             def single_predict_mean(args):
#                 Fmu, Fvar = args
#                 integrand2 = lambda *X: self.likelihood.conditional_variance(
#                     *X
#                 ) + tf.square(self.likelihood.conditional_mean(*X))
#                 epsilon = None
#                 E_y, E_y2 = ndiag_mc(
#                     [self.likelihood.conditional_mean, integrand2],
#                     S=self.likelihood.num_monte_carlo_points,
#                     Fmu=Fmu,
#                     Fvar=Fvar,
#                     epsilon=epsilon,
#                 )
#                 return E_y

#             if num_inducing_samples is None:
#                 mixing_probs = self.likelihood.predict_mean_and_var(Fmu, Fvar)[0]
#             else:
#                 # mixing_probs = tf.map_fn(single_predict_mean, (Fmu, Fvar),
#                 #                          dtype=tf.float64)
#                 mixing_probs = tf.vectorized_map(single_predict_mean, (Fmu, Fvar))
#             return mixing_probs

#         # mixing_probs = tf.map_fn(single_output_predict_mean, (Fmu, Fvar),
#         #                          dtype=tf.float64)
#         mixing_probs = tf.vectorized_map(single_output_predict_mean, (Fmu, Fvar))
#         if num_inducing_samples is None:
#             mixing_probs = tf.transpose(mixing_probs, [1, 0, 2])
#         else:
#             mixing_probs = tf.transpose(mixing_probs, [1, 2, 0, 3])
#         # mixing_probs = tf.transpose(mixing_probs, [1, 2, 0, 3])
#         return mixing_probs


# class SVGPGatingNetworkBinary(SVGPGatingNetworkBase):
#     def __init__(
#         self, gating_function: SVGPGatingFunction = None, name="GatingNetwork"
#     ):
#         assert isinstance(gating_function, SVGPGatingFunction)
#         gating_function_list = [gating_function]
#         super().__init__(gating_function_list, name=name)
#         # self.gating_function = gating_function
#         self.likelihood = Bernoulli()
#         self.num_experts = 2

#     # def prior_kls(self) -> tf.Tensor:
#     #     """Returns the set of experts KL divergences as a batched tensor.

#     #     :returns: a Tensor with shape [num_experts,]
#     #     """
#     #     return tf.convert_to_tensor(self.gating_function.prior_kl())

#     def predict_fs(
#         self, Xnew: InputData, num_inducing_samples: int = None
#     ) -> MeanAndVariance:
#         f_mu, f_var = self.gating_function_list[0].predict_f(Xnew, num_inducing_samples)
#         Fmu = tf.stack([f_mu, -f_mu], -1)
#         Fvar = tf.stack([f_var, f_var], -1)
#         return Fmu, Fvar

#     def predict_mixing_probs(self, Xnew: InputData, num_inducing_samples: int = None):
#         """Compute mixing probabilities.

#         Returns a tensor with dimensions,
#             [num_inducing_samples,num_data, output_dim, num_experts]
#         if num_inducing_samples=None otherwise a tensor with dimensions,
#             [num_data, output_dim, num_experts]

#         .. math::
#             \\mathbf{u}_h \sim \mathcal{N}(q\_mu, q\_sqrt \cdot q\_sqrt^T) \\\\
#             \\Pr(\\alpha=k | \\mathbf{Xnew}, \\mathbf{u}_h)

#         :param Xnew: test input(s) [num_data, input_dim]
#         :param num_inducing_samples: how many samples to draw from inducing points
#         """
#         h_mu, h_var = self.gating_function_list[0].predict_f(
#             Xnew, num_inducing_samples, full_cov=False
#         )

#         def single_predict_mean(args):
#             h_mu, h_var = args
#             return self.likelihood.predict_mean_and_var(h_mu, h_var)[0]

#         if num_inducing_samples is None:
#             prob_a_0 = self.likelihood.predict_mean_and_var(h_mu, h_var)[0]
#         else:
#             prob_a_0 = tf.map_fn(single_predict_mean, (h_mu, h_var), dtype=tf.float64)

#         prob_a_1 = 1 - prob_a_0
#         mixing_probs = tf.stack([prob_a_0, prob_a_1], -1)
#         return mixing_probs
