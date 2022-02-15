#!/usr/bin/env python3
from abc import abstractmethod
from typing import Optional, Union

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.inducing_variables import INDUCING_VARIABLE_OBJECTS, InducingVariables
from gpflow.kernels import KERNEL_OBJECTS, Kernel, MultioutputKernel
from gpflow.mean_functions import MEAN_FUNCTION_OBJECTS, MeanFunction
from gpflow.models.model import GPModel
from gpflow.utilities.keras import try_array_except_none, try_val_except_none
from mogpe.custom_types import (
    ExpertIndicatorCategoricalDist,
    GatingFunctionSamples,
    GatingMeanAndVariance,
    InputData,
    MixingProb,
    MixingProbSamples,
    NumData,
    NumGatingFunctions,
    NumSamples,
)

from .gps import predict_f_given_inducing_samples

tfd = tfp.distributions


class GatingNetworkBase(gpf.Module):
    """Interface for gating networks"""

    def __init__(self, num_experts: int, name: Optional[str] = "gating_network"):
        super().__init__(name=name)
        self._num_experts = num_experts

    def predict_categorical_dist(self, Xnew: InputData, **kwargs) -> tfd.Categorical:
        r"""Return probability mass function over expert indicator as Categorical dist

        .. math::
            P(\alpha | Xnew)
        batch_shape == NumData
        """
        mixing_probs = self.predict_mixing_probs(Xnew, **kwargs)
        return tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")

    @abstractmethod
    def predict_mixing_probs(self, Xnew: InputData, **kwargs) -> MixingProb:
        r"""Calculates the set of experts mixing probabilities at Xnew

        :math:`\{\Pr(\\alpha=k | x)\}^K_{k=1}`
        """
        raise NotImplementedError

    @property
    def num_experts(self):
        return self._num_experts


class GPGatingNetworkBase(GatingNetworkBase):
    """Interface for GP-based gating network"""

    def __init__(self, gp: GPModel, num_experts: int, name: str = "gp_gating_network"):
        super().__init__(num_experts=num_experts, name=name)
        self._gp = gp

    @abstractmethod
    def predict_mixing_probs(self, Xnew: InputData, **kwargs) -> MixingProb:
        r"""Calculates the set of experts mixing probabilities at Xnew

        :math:`\{\Pr(\\alpha=k | x)\}^K_{k=1}`
        """
        raise NotImplementedError

    def predict_f(
        self, Xnew: InputData, full_cov: Optional[bool] = True
    ) -> GatingMeanAndVariance:
        """Calculates the set of gating function GP posteriors at Xnew"""
        return self.gp.predict_f(Xnew, full_cov=full_cov)

    @property
    def gp(self):
        return self._gp


class SVGPGatingNetwork(GPGatingNetworkBase):
    def __init__(
        self,
        kernel: Kernel,
        inducing_variable: InducingVariables,
        mean_function: MeanFunction = None,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        name: Optional[str] = "SVGPGatingNetwork",
    ):
        if isinstance(kernel, MultioutputKernel):
            self.num_gating_gps = kernel.num_latent_gps
            num_experts = self.num_gating_gps
            likelihood = gpf.likelihoods.Softmax(num_classes=self.num_gating_gps)
        else:
            self.num_gating_gps = 1
            num_experts = 2
            likelihood = gpf.likelihoods.Bernoulli()

        svgp = gpf.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            num_latent_gps=self.num_gating_gps,
            q_diag=q_diag,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=whiten,
            num_data=None,
        )

        super().__init__(gp=svgp, num_experts=num_experts)

    def call(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = 1,
        training: Optional[bool] = False,
    ) -> Union[MixingProbSamples, ExpertIndicatorCategoricalDist]:
        # TODO move this to training
        if training:
            return self.predict_categorical_dist_given_h_samples(
                Xnew, num_h_samples=num_samples
            )  # [S, N, K]
        else:
            return self.predict_categorical_dist(Xnew)

    def predict_categorical_dist(self, Xnew: InputData) -> tfd.Categorical:
        r"""Return probability mass function over expert indicator as Categorical dist

        :return: categorical dist with batch_shape [NumData]
        """
        mixing_probs = self.predict_mixing_probs(Xnew)
        return tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")

    def predict_categorical_dist_given_h_samples(
        self, Xnew: InputData, num_h_samples: Optional[int] = 1
    ) -> tfd.Categorical:
        r"""Return probability mass function over expert indicator as Categorical dist

        :return: categorical dist with batch_shape [NumSamples, NumData]
        """
        h_samples = self.predict_h_samples(
            Xnew, num_samples=num_h_samples, full_cov=False
        )
        mixing_probs = self.predict_mixing_probs_given_h(h_samples)  # [S, N, K]
        cat = tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")
        return tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")

    def predict_categorical_dist_given_inducing_samples(
        self, Xnew: InputData, num_inducing_samples: Optional[int] = 1
    ) -> tfd.Categorical:
        r"""Return probability mass function over expert indicator as Categorical dist

        :return: categorical dist with batch_shape [NumSamples, NumData]
        """
        # mixing_probs = self.predict_mixing_probs(Xnew, num_inducing_samples=nu
        h_mean, h_var = self.predict_h(
            Xnew, num_inducing_samples=num_inducing_samples, full_cov=False
        )
        mixing_probs = self.predict_mixing_probs_given_h(h_mean, h_var)
        return tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")

    def predict_mixing_probs(
        self,
        Xnew: InputData,
        num_inducing_samples: Optional[int] = None,
        full_cov: Optional[bool] = False,
    ) -> Union[MixingProb, MixingProbSamples]:
        r"""Compute mixing probabilities at Xnew

        Pr(\alpha = k | Xnew) = \int Pr(\alpha = k | h) p(h | Xnew) dh
        """
        h_mean, h_var = self.predict_h(
            Xnew, num_inducing_samples=num_inducing_samples, full_cov=full_cov
        )
        return self.predict_mixing_probs_given_h(h_mean, h_var)

    def predict_h_samples(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = None,
        full_cov: Optional[bool] = False,
    ) -> GatingFunctionSamples:
        h_samples = self.gp.predict_f_samples(
            Xnew, num_samples=num_samples, full_cov=full_cov
        )
        if self.num_gating_gps == 1:
            h_samples = tf.concat([h_samples, -h_samples], -1)
        return h_samples

    def predict_h(
        self,
        Xnew: InputData,
        num_inducing_samples: Optional[int] = None,
        full_cov: Optional[bool] = False,
    ) -> GatingMeanAndVariance:
        if num_inducing_samples:
            h_mean, h_var = predict_f_given_inducing_samples(
                Xnew=Xnew,
                svgp=self.gp,
                num_inducing_samples=num_inducing_samples,
                full_cov=full_cov,
            )
        else:
            h_mean, h_var = self.gp.predict_f(Xnew, full_cov=full_cov)

        # h_mean, h_var = self.gp.predict_f(
        #     Xnew, num_inducing_samples=num_inducing_samples, full_cov=full_cov
        # )
        if self.num_gating_gps == 1:
            h_mean = tf.concat([h_mean, -h_mean], -1)
            if full_cov:
                h_var = tf.concat([h_var, h_var], 0)
            else:
                h_var = tf.concat([h_var, h_var], -1)
        return h_mean, h_var

    def predict_mixing_probs_given_h(
        self,
        h_mean: Union[
            ttf.Tensor2[NumData, NumGatingFunctions],
            ttf.Tensor3[NumSamples, NumData, NumGatingFunctions],
        ],
        h_var: Optional[ttf.Tensor2[NumData, NumGatingFunctions]] = None,
    ) -> Union[MixingProb, MixingProbSamples]:
        r"""Compute mixing probabilities given gating functions.

        if h_vars is None:
            Pr(\alpha = k | h) where h=h_mean
        else:
            \int Pr(\alpha = k | h) N(h | h_mean, h_var) dh
        """
        if h_var is not None:
            mixing_probs = self.gp.likelihood.predict_mean_and_var(h_mean, h_var)[0]
        else:
            mixing_probs = self.gp.likelihood.conditional_mean(h_mean)
        return mixing_probs

    def prior_kl(self) -> ttf.Tensor1[NumGatingFunctions]:
        """Returns the gating functions' KL divergence(s)"""
        return self.gp.prior_kl()

    @property
    def gp(self):
        return self._gp

    def get_config(self):
        return {
            "kernel": tf.keras.layers.serialize(self.gp.kernel),
            "inducing_variable": tf.keras.layers.serialize(self.gp.inducing_variable),
            "mean_function": tf.keras.layers.serialize(self.gp.mean_function),
            "q_diag": self.gp.q_diag,
            "q_mu": self.gp.q_mu.numpy(),
            "q_sqrt": self.gp.q_sqrt.numpy(),
            "whiten": self.gp.whiten,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        kernel = tf.keras.layers.deserialize(
            cfg["kernel"], custom_objects=KERNEL_OBJECTS
        )
        inducing_variable = tf.keras.layers.deserialize(
            cfg["inducing_variable"], custom_objects=INDUCING_VARIABLE_OBJECTS
        )
        try:
            mean_function = tf.keras.layers.deserialize(
                cfg["mean_function"], custom_objects=MEAN_FUNCTION_OBJECTS
            )
        except KeyError:
            mean_function = gpf.mean_functions.Zero()
        return cls(
            kernel=kernel,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            q_diag=try_val_except_none(cfg, "q_diag"),
            q_mu=try_array_except_none(cfg, "q_mu"),
            q_sqrt=try_array_except_none(cfg, "q_sqrt"),
            whiten=try_val_except_none(cfg, "whiten"),
        )


# class SVGPGatingNetwork(GPGatingNetworkBase):
#     def __init__(self, svgp: SVGPPrior):
#         self.num_gating_gps = svgp.num_latent_gps
#         if self.num_gating_gps == 1:
#             self._likelihood = gpf.likelihoods.Bernoulli()
#             num_experts = 2
#         else:
#             self._likelihood = gpf.likelihoods.Softmax(num_classes=self.num_gating_gps)
#             num_experts = self.num_gating_gps
#         super().__init__(gp=svgp, num_experts=num_experts)

#     def call(
#         self,
#         Xnew: InputData,
#         num_samples: Optional[int] = 1,
#         training: Optional[bool] = False,
#     ) -> Union[MixingProbSamples, ExpertIndicatorCategoricalDist]:
#         # TODO move this to training
#         if training:
#             return self.predict_categorical_dist_given_h_samples(
#                 Xnew, num_h_samples=num_samples
#             )  # [S, N, K]
#         else:
#             return self.predict_categorical_dist(Xnew)

#     def predict_categorical_dist(self, Xnew: InputData) -> tfd.Categorical:
#         r"""Return probability mass function over expert indicator as Categorical dist

#         :return: categorical dist with batch_shape [NumData]
#         """
#         mixing_probs = self.predict_mixing_probs(Xnew)
#         print("inside predict_categorical_dist")
#         print(mixing_probs.shape)
#         return tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")

#     def predict_categorical_dist_given_h_samples(
#         self, Xnew: InputData, num_h_samples: Optional[int] = 1
#     ) -> tfd.Categorical:
#         r"""Return probability mass function over expert indicator as Categorical dist

#         :return: categorical dist with batch_shape [NumSamples, NumData]
#         """
#         h_samples = self.predict_h_samples(
#             Xnew, num_samples=num_h_samples, full_cov=False
#         )
#         mixing_probs = self.predict_mixing_probs_given_h(h_samples)  # [S, N, K]
#         print("inside predict_categorical_dist_given_h_samples")
#         print(mixing_probs.shape)
#         cat = tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")
#         print("cat")
#         print(cat)
#         return tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")

#     def predict_categorical_dist_given_inducing_samples(
#         self, Xnew: InputData, num_inducing_samples: Optional[int] = 1
#     ) -> tfd.Categorical:
#         r"""Return probability mass function over expert indicator as Categorical dist

#         :return: categorical dist with batch_shape [NumSamples, NumData]
#         """
#         # mixing_probs = self.predict_mixing_probs(Xnew, num_inducing_samples=nu
#         h_mean, h_var = self.predict_h(
#             Xnew, num_inducing_samples=num_inducing_samples, full_cov=False
#         )
#         mixing_probs = self.predict_mixing_probs_given_h(h_mean, h_var)
#         print("inside predict_categorical_dist_given_inducing_samples")
#         print(mixing_probs.shape)
#         return tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")

#     def predict_mixing_probs(
#         self,
#         Xnew: InputData,
#         num_inducing_samples: Optional[int] = None,
#         full_cov: Optional[bool] = False,
#     ) -> Union[MixingProb, MixingProbSamples]:
#         r"""Compute mixing probabilities at Xnew

#         Pr(\alpha = k | Xnew) = \int Pr(\alpha = k | h) p(h | Xnew) dh
#         """
#         h_mean, h_var = self.predict_h(
#             Xnew, num_inducing_samples=num_inducing_samples, full_cov=full_cov
#         )
#         return self.predict_mixing_probs_given_h(h_mean, h_var)

#     def predict_h_samples(
#         self,
#         Xnew: InputData,
#         num_samples: Optional[int] = None,
#         full_cov: Optional[bool] = False,
#     ) -> GatingFunctionSamples:
#         h_samples = self.gp.predict_f_samples(
#             Xnew, num_samples=num_samples, full_cov=full_cov
#         )
#         if self.num_gating_gps == 1:
#             h_samples = tf.concat([h_samples, -h_samples], -1)
#         return h_samples

#     def predict_h(
#         self,
#         Xnew: InputData,
#         num_inducing_samples: Optional[int] = None,
#         full_cov: Optional[bool] = False,
#     ) -> GatingMeanAndVariance:
#         if num_inducing_samples:
#             h_mean, h_var = predict_f_given_inducing_samples(
#                 Xnew=Xnew,
#                 svgp=self.gp,
#                 num_inducing_samples=num_inducing_samples,
#                 full_cov=full_cov,
#             )
#         else:
#             h_mean, h_var = self.gp.predict_f(Xnew, full_cov=full_cov)

#         # h_mean, h_var = self.gp.predict_f(
#         #     Xnew, num_inducing_samples=num_inducing_samples, full_cov=full_cov
#         # )
#         if self.num_gating_gps == 1:
#             h_mean = tf.concat([h_mean, -h_mean], -1)
#             if full_cov:
#                 h_var = tf.concat([h_var, h_var], 0)
#             else:
#                 h_var = tf.concat([h_var, h_var], -1)
#         return h_mean, h_var

#     def predict_mixing_probs_given_h(
#         self,
#         h_mean: Union[
#             ttf.Tensor2[NumData, NumGatingFunctions],
#             ttf.Tensor3[NumSamples, NumData, NumGatingFunctions],
#         ],
#         h_var: Optional[ttf.Tensor2[NumData, NumGatingFunctions]] = None,
#     ) -> Union[MixingProb, MixingProbSamples]:
#         r"""Compute mixing probabilities given gating functions.

#         if h_vars is None:
#             Pr(\alpha = k | h) where h=h_mean
#         else:
#             \int Pr(\alpha = k | h) N(h | h_mean, h_var) dh
#         """
#         if h_var is not None:
#             mixing_probs = self.likelihood.predict_mean_and_var(h_mean, h_var)[0]
#         else:
#             mixing_probs = self.likelihood.conditional_mean(h_mean)
#         return mixing_probs

#     def prior_kl(self) -> ttf.Tensor1[NumGatingFunctions]:
#         """Returns the gating functions' KL divergence(s)"""
#         return self.gp.prior_kl()

#     @property
#     def gp(self):
#         return self._gp

#     @property
#     def likelihood(self):
#         return self._likelihood

#     def get_config(self):
#         return {"svgp": tf.keras.layers.serialize(self.gp)}

#     @classmethod
#     def from_config(cls, cfg: dict):
#         svgp_layer = tf.keras.layers.deserialize(
#             cfg["svgp"], custom_objects={"SVGPPrior": SVGPPrior}
#         )
#         return cls(svgp_layer)
