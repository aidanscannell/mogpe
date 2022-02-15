#!/usr/bin/env python3
from typing import Optional

import gpflow as gpf
import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import posteriors
from gpflow.conditionals import conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float
from gpflow.inducing_variables import INDUCING_VARIABLE_OBJECTS
from gpflow.kernels import KERNEL_OBJECTS, Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MEAN_FUNCTION_OBJECTS, MeanFunction
from gpflow.models import SVGP
from mogpe.custom_types import (
    InputData,
    MeanAndVariance,
    NumData,
    NumSamples,
    OutputDim,
)

tfd = tfp.distributions


def predict_f_given_inducing_samples(
    Xnew: InputData,
    svgp: SVGP,
    num_inducing_samples: int = 1,
    full_cov: Optional[bool] = False,
):
    q_mu = tf.transpose(svgp.q_mu, [1, 0])
    q_dist = tfp.distributions.MultivariateNormalTriL(
        loc=q_mu,
        scale_tril=svgp.q_sqrt,
        validate_args=False,
        allow_nan_stats=True,
        name="InducingOutputMultivariateNormalQ",
    )
    inducing_samples = q_dist.sample(num_inducing_samples)
    q_mu = tf.transpose(inducing_samples, [0, 2, 1])

    @tf.function
    def single_sample_conditional(q_mu):
        return conditional(
            Xnew,
            svgp.inducing_variable,
            svgp.kernel,
            q_mu,
            q_sqrt=None,
            full_cov=full_cov,
            white=svgp.whiten,
            full_output_cov=False,
        )

    mean, var = tf.map_fn(
        single_sample_conditional,
        q_mu,
        fn_output_signature=(default_float(), default_float()),
    )

    return mean + svgp.mean_function(Xnew), var


# def sample_inducing_points(
#     svgp: SVGP, num_samples: int = None
# ) -> ttf.Tensor3[NumSamples, NumData, OutputDim]:
#     """Returns samples from the inducing point distribution.

#     .. math::
#         q \sim \mathcal{N}(q\_mu, q\_sqrt q\_sqrt^T)
#     """


# class SVGPPrior(tf.keras.layers.Layer):
#     """SVGP prior with inducing point sampling.

#     predict_f have an argument to set the number of samples
#     (num_inducing_samples) to draw form the inducing outputs distribution.
#     If num_inducing_samples is None then the standard functionality is achieved, i.e. the inducing points
#     are marginalised in closed-form.
#     If num_inducing_samples=3 then 3 samples are drawn from the inducing ouput distribution and the standard
#     GP conditional is called (q_sqrt=None). The results for each sample are stacked on the leading dimension
#     and the user now has the ability marginalise them outside of this class.
#     """

#     def __init__(
#         self,
#         kernel: Kernel,
#         inducing_variable,
#         # likelihood: Likelihood = None,
#         mean_function: MeanFunction = None,
#         num_latent_gps: int = 1,
#         q_diag: bool = False,
#         q_mu=None,
#         q_sqrt=None,
#         whiten: bool = True,
#         name: Optional[str] = "SVGPPrior",
#     ):
#         super().__init__(name=name)
#         self._gp = gpf.models.SVGP(
#             kernel=kernel,
#             # likelihood=Likelihood,
#             likelihood=None,
#             inducing_variable=inducing_variable,
#             mean_function=mean_function,
#             num_latent_gps=num_latent_gps,
#             q_diag=q_diag,
#             q_mu=q_mu,
#             q_sqrt=q_sqrt,
#             whiten=whiten,
#             num_data=None,
#         )

#     def call(
#         self,
#         Xnew: InputData,
#         num_inducing_samples: int = None,
#         full_cov: bool = False,
#         full_output_cov: bool = False,
#         training: Optional[bool] = False,
#     ):
#         return self.predict_f(
#             Xnew,
#             num_inducing_samples=num_inducing_samples,
#             full_cov=full_cov,
#             full_output_cov=full_output_cov,
#         )

#     def predict_f(
#         self,
#         Xnew: InputData,
#         num_inducing_samples: int = None,
#         full_cov: bool = False,
#         full_output_cov: bool = False,
#     ) -> MeanAndVariance:
#         """Compute mean and (co)variance of latent function at Xnew.

#         If num_inducing_samples is not None then sample inducing points instead
#         of integrating them in closed-form. This is required in the mixture of
#         experts lower bound.

#         :param Xnew: inputs with shape [num_test, input_dim]
#         :param num_inducing_samples:
#             number of samples to draw from inducing points distribution.
#         :param full_cov:
#             If True, draw correlated samples over Xnew. Computes the Cholesky over the
#             dense covariance matrix of size [num_data, num_data].
#             If False, draw samples that are uncorrelated over the inputs.
#         :param full_output_cov:
#             If True, draw correlated samples over the outputs.
#             If False, draw samples that are uncorrelated over the outputs.
#         :returns: tuple of Tensors (mean, variance),
#             If num_inducing_samples=None,
#                 means.shape == [num_test, output_dim],
#                 If full_cov=True and full_output_cov=False,
#                     var.shape == [output_dim, num_test, num_test]
#                 If full_cov=False,
#                     var.shape == [num_test, output_dim]
#             If num_inducing_samples is not None,
#                 means.shape == [num_inducing_samples, num_test, output_dim],
#                 If full_cov=True and full_output_cov=False,
#                     var.shape == [num_inducing_samples, output_dim, num_test, num_test]
#                 If full_cov=False and full_output_cov=False,
#                     var.shape == [num_inducing_samples, num_test, output_dim]
#         """
#         with tf.name_scope("predict_f") as scope:
#             if num_inducing_samples is None:
#                 return self.posterior(
#                     posteriors.PrecomputeCacheType.NOCACHE
#                 ).fused_predict_f(
#                     Xnew, full_cov=full_cov, full_output_cov=full_output_cov
#                 )
#             else:
#                 q_mu = self.sample_inducing_points(num_inducing_samples)

#                 @tf.function
#                 def single_sample_conditional(q_mu):
#                     return conditional(
#                         Xnew,
#                         self.inducing_variable,
#                         self.kernel,
#                         q_mu,
#                         q_sqrt=None,
#                         full_cov=full_cov,
#                         white=self.whiten,
#                         full_output_cov=full_output_cov,
#                     )

#                 mean, var = tf.map_fn(
#                     single_sample_conditional,
#                     q_mu,
#                     fn_output_signature=(default_float(), default_float()),
#                 )
#             return mean + self.mean_function(Xnew), var

#     def sample_inducing_points(
#         self, num_samples: int = None
#     ) -> ttf.Tensor3[NumSamples, NumData, OutputDim]:
#         """Returns samples from the inducing point distribution.

#         .. math::
#             q \sim \mathcal{N}(q\_mu, q\_sqrt q\_sqrt^T)
#         """
#         q_mu = tf.transpose(self.q_mu, [1, 0])
#         q_dist = tfp.distributions.MultivariateNormalTriL(
#             loc=q_mu,
#             scale_tril=self.q_sqrt,
#             validate_args=False,
#             allow_nan_stats=True,
#             name="InducingOutputMultivariateNormalQ",
#         )
#         inducing_samples = q_dist.sample(num_samples)
#         return tf.transpose(inducing_samples, [0, 2, 1])

#     # def predict_f_samples(
#     #     self,
#     #     Xnew: InputData,
#     #     num_samples: Optional[int] = None,
#     #     full_cov: Optional[bool] = False,
#     # ) -> MeanAndVariance:
#     #     return self.gp.predict_f_samples(
#     #         Xnew, num_samples=num_samples, full_cov=full_cov
#     #     )

#     def predict_f_samples(
#         self,
#         Xnew: InputData,
#         num_samples: Optional[int] = None,
#         full_cov: bool = True,
#         full_output_cov: bool = False,
#     ) -> tf.Tensor:
#         """Samples the posterior latent function(s) at Xnew."""
#         print("inside svgpprior predict_f_samples")
#         if full_cov and full_output_cov:
#             raise NotImplementedError(
#                 "The combination of both `full_cov` and `full_output_cov` is not supported."
#             )

#         # check below for shape info
#         mean, cov = self.predict_f(
#             Xnew, full_cov=full_cov, full_output_cov=full_output_cov
#         )
#         if full_cov:
#             # mean: [..., N, P]
#             # cov: [..., P, N, N]
#             mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
#             samples = sample_mvn(
#                 mean_for_sample, cov, full_cov, num_samples=num_samples
#             )  # [..., (S), P, N]
#             samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
#         else:
#             # mean: [..., N, P]
#             # cov: [..., N, P] or [..., N, P, P]
#             samples = sample_mvn(
#                 mean, cov, full_output_cov, num_samples=num_samples
#             )  # [..., (S), N, P]
#         return samples  # [..., (S), N, P]

#     def prior_kl(self):
#         return self.gp.prior_kl()

#     @property
#     def gp(self):
#         return self._gp

#     @property
#     def posterior(self):
#         return self.gp.posterior

#     @property
#     def kernel(self):
#         return self.gp.kernel

#     @property
#     def inducing_variable(self):
#         return self.gp.inducing_variable

#     @property
#     def mean_function(self):
#         return self.gp.mean_function

#     @property
#     def num_latent_gps(self):
#         return self.gp.num_latent_gps

#     @property
#     def q_diag(self):
#         return self.gp.q_diag

#     @property
#     def q_mu(self):
#         return self.gp.q_mu

#     @property
#     def q_sqrt(self):
#         return self.gp.q_sqrt

#     @property
#     def whiten(self):
#         return self.gp.whiten

#     def get_config(self):
#         return {
#             "kernel": tf.keras.layers.serialize(self.kernel),
#             "mean_function": tf.keras.layers.serialize(self.mean_function),
#             "inducing_variable": tf.keras.layers.serialize(self.inducing_variable),
#             "num_latent_gps": self.num_latent_gps,
#             "q_diag": self.q_diag,
#             "q_mu": self.q_mu.numpy(),
#             "q_sqrt": self.q_sqrt.numpy(),
#             "whiten": self.whiten,
#         }

#     @classmethod
#     def from_config(cls, cfg: dict):
#         kernel = tf.keras.layers.deserialize(
#             cfg["kernel"], custom_objects=KERNEL_OBJECTS
#         )
#         mean_function = tf.keras.layers.deserialize(
#             cfg["mean_function"], custom_objects=MEAN_FUNCTION_OBJECTS
#         )
#         inducing_variable = tf.keras.layers.deserialize(
#             cfg["inducing_variable"], custom_objects=INDUCING_VARIABLE_OBJECTS
#         )
#         return cls(
#             kernel=kernel,
#             mean_function=mean_function,
#             inducing_variable=inducing_variable,
#             num_latent_gps=try_val_except_none(cfg, "num_latent_gps"),
#             q_diag=try_val_except_none(cfg, "q_diag"),
#             q_mu=try_array_except_none(cfg, "q_mu"),
#             q_sqrt=try_array_except_none(cfg, "q_sqrt"),
#             whiten=try_val_except_none(cfg, "whiten"),
#         )


# def try_array_except_none(cfg: dict, key: str):
#     # np.array(cfg["q_mu"]) works for deserializing model using keras
#     # and setting q_mu=None/not setting them, allows users to write custom configs
#     # without specifying q_mu/q_sqrt in the config
#     try:
#         return np.array(cfg[key]) if cfg[key] is not None else None
#     except KeyError:
#         return None


# def try_val_except_none(cfg: dict, key: str):
#     try:
#         return cfg[key]
#     except KeyError:
#         return None
