#!/usr/bin/env python3
import abc
from typing import Optional, Union

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.inducing_variables import INDUCING_VARIABLE_OBJECTS, InducingVariables
from gpflow.kernels import KERNEL_OBJECTS, Kernel
from gpflow.likelihoods import LIKELIHOOD_OBJECTS, Likelihood
from gpflow.mean_functions import MEAN_FUNCTION_OBJECTS, MeanFunction
from gpflow.models import SVGP
from gpflow.utilities.keras import try_array_except_none, try_val_except_none
from gpflow.models.model import MeanAndVariance
from mogpe.custom_types import InputData, NumExperts

from .gps import predict_f_given_inducing_samples

tfd = tfp.distributions

ExpertDist = Union[tfd.MultivariateNormalFullCovariance, tfd.MultivariateNormalDiag]


class ExpertBase(gpf.Module):
    """Interface for individual expert."""

    @abc.abstractmethod
    def predict_dist(self, Xnew: InputData, **kwargs) -> tfd.Distribution:
        """Returns the individual experts prediction at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: an instance of a TensorFlow Distribution
        """
        raise NotImplementedError


class SVGPExpert(ExpertBase):
    """Sparse Variational Gaussian Process Expert."""

    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable: InducingVariables,
        mean_function: MeanFunction = None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        name: Optional[str] = "SVGPExpert",
    ):
        svgp = SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
            q_diag=q_diag,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=whiten,
            num_data=None,
        )
        super().__init__(name=name)
        self._gp = svgp

    def call(
        self,
        Xnew: InputData,
        num_inducing_samples: Optional[int] = 1,
        training: Optional[bool] = False,
    ) -> ExpertDist:
        """
        :returns: instance of tfd.MultivariateNormalDiag
            event_shape == [OutputDim]
            if training
                batch_shape == [NumSamples, NumData]
            else:
                batch_shape == [NumData]
        """
        if training:
            return self.predict_dist_given_inducing_samples(
                Xnew, num_inducing_samples=num_inducing_samples, full_cov=False
            )
        else:
            return self.predict_dist(Xnew)

    def predict_dist(self, Xnew: InputData, full_cov: bool = False) -> ExpertDist:
        """Returns the mean and (co)variance of the expert's prediction at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
        """
        y_mean, y_var = self.predict_y(Xnew, full_cov=full_cov)
        return self.predict_dist_given_y(y_mean, y_var)

    def predict_dist_given_inducing_samples(
        self,
        Xnew: InputData,
        num_inducing_samples: int = None,
        full_cov: bool = False,
    ) -> ExpertDist:
        """Returns a tf.Distribution for the expert's prediction at Xnew.

        Used in the tight lower bound

        :param num_inducing_samples:
            the number of samples to draw from the inducing points joint distribution.
        """
        f_mean, f_var = self.predict_f(
            Xnew,
            num_inducing_samples=num_inducing_samples,
            full_cov=full_cov,
        )
        y_mean, y_var = self.gp.likelihood((f_mean, f_var))
        return self.predict_dist_given_y(y_mean, y_var)

    def predict_dist_given_f_samples(
        self,
        Xnew: InputData,
        num_f_samples: int = None,
        full_cov: bool = False,
    ) -> ExpertDist:
        """Returns a tf.Distribution for the expert's prediction at Xnew.

        Used in the further lower bound.
        It changes the likelihood approximation and results in worse performance.

        :param num_f_samples:
            the number of samples to draw from the standard SVGP variational posterior
        """
        f_samples = self.gp.predict_f_samples(
            Xnew,
            num_samples=num_f_samples,
            full_cov=full_cov,
        )
        # TODO not working for multioutput Gaussian likelihood
        y_var = self.gp.likelihood.conditional_variance(f_samples)
        return self.predict_dist_given_y(f_samples, y_var)

    def predict_y(
        self,
        Xnew: InputData,
        num_inducing_samples: int = None,
        full_cov: bool = False,
    ) -> MeanAndVariance:
        """Returns the mean and (co)variance of the experts' latent function at Xnew."""
        f_mean, f_var = self.predict_f(
            Xnew,
            num_inducing_samples=num_inducing_samples,
            full_cov=full_cov,
        )
        return self.gp.likelihood((f_mean, f_var))

    def predict_dist_given_y(self, y_mean, y_var) -> ExpertDist:
        """Returns the mean and (co)variance of the expert's prediction at Xnew.

        batch_shape [..., NumData]
        event_shape [OutputDim]"""
        # return tfd.MultivariateNormalDiag(loc=y_mean, scale_diag=y_var ** 2)
        # return tfd.MultivariateNormalDiag(loc=y_mean, scale_diag=tf.math.sqrt(y_var))
        return tfd.MultivariateNormalDiag(loc=y_mean, scale_diag=y_var)
        # if y_mean.shape == y_var.shape:  # full_cov==False
        #     return tfd.MultivariateNormalDiag(loc=y_mean, scale_diag=y_var ** 2)
        # else:
        # return tfd.MultivariateNormalTriL(
        #     loc=y_mean, scale_tril=tf.linalg.cholesky(y_var ** 2)
        # )
        # return tfd.MultivariateNormalFullCovariance(
        #     loc=y_mean, covariance_matrix=y_var ** 2
        # )

    def predict_f(
        self,
        Xnew: InputData,
        num_inducing_samples: Optional[int] = None,
        full_cov: Optional[bool] = False,
    ) -> MeanAndVariance:
        if num_inducing_samples:
            return predict_f_given_inducing_samples(
                Xnew=Xnew,
                svgp=self.gp,
                num_inducing_samples=num_inducing_samples,
                full_cov=full_cov,
            )
        else:
            return self.gp.predict_f(Xnew, full_cov=full_cov)

    # def predict_f_samples(
    #     self,
    #     Xnew: InputData,
    #     num_samples: Optional[int] = None,
    #     full_cov: Optional[bool] = False,
    # ) -> ttf.Tensor3[NumSamples, NumData, OutputDim]:
    #     print("inside predict_f_samples")
    #     return self.gp.predict_f_samples(
    #         Xnew, num_samples=num_samples, full_cov=full_cov
    #     )

    def prior_kl(self) -> ttf.Tensor1[NumExperts]:
        """Returns the expert's KL divergence"""
        return self.gp.prior_kl()

    @property
    def gp(self):
        return self._gp

    def get_config(self):
        return {
            "kernel": tf.keras.layers.serialize(self.gp.kernel),
            "likelihood": tf.keras.layers.serialize(self.gp.likelihood),
            "inducing_variable": tf.keras.layers.serialize(self.gp.inducing_variable),
            "mean_function": tf.keras.layers.serialize(self.gp.mean_function),
            "num_latent_gps": self.gp.num_latent_gps,
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
        likelihood = tf.keras.layers.deserialize(
            cfg["likelihood"], custom_objects=LIKELIHOOD_OBJECTS
        )
        inducing_variable = tf.keras.layers.deserialize(
            cfg["inducing_variable"], custom_objects=INDUCING_VARIABLE_OBJECTS
        )
        mean_function = tf.keras.layers.deserialize(
            cfg["mean_function"], custom_objects=MEAN_FUNCTION_OBJECTS
        )
        return cls(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            num_latent_gps=try_val_except_none(cfg, "num_latent_gps"),
            q_diag=try_val_except_none(cfg, "q_diag"),
            q_mu=try_array_except_none(cfg, "q_mu"),
            q_sqrt=try_array_except_none(cfg, "q_sqrt"),
            whiten=try_val_except_none(cfg, "whiten"),
        )
