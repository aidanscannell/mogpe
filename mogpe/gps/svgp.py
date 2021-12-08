#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import posteriors
from gpflow.conditionals import conditional
from gpflow.config import default_float
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models import SVGP
from gpflow.models.model import InputData, MeanAndVariance

tfd = tfp.distributions


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
        num_data=None,
    ):
        super().__init__(
            kernel,
            likelihood,
            inducing_variable,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
            q_diag=q_diag,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=whiten,
            num_data=num_data,
        )

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
            name="InducingOutputMultivariateNormalQ",
        )
        inducing_samples = q_dist.sample(num_samples)
        return tf.transpose(inducing_samples, [0, 2, 1])

    def predict_f(
        self,
        Xnew: InputData,
        num_inducing_samples: int = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """Compute mean and (co)variance of latent function at Xnew.

        If num_inducing_samples is not None then sample inducing points instead
        of analytically integrating them. This is required in the mixture of
        experts lower bound.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param num_inducing_samples:
            number of samples to draw from inducing points distribution.
        :param full_cov:
            If True, draw correlated samples over Xnew. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.
        :returns: tuple of Tensors (mean, variance),
            If num_inducing_samples=None,
                means.shape == [num_test, output_dim],
                If full_cov=True and full_output_cov=False,
                    var.shape == [output_dim, num_test, num_test]
                If full_cov=False,
                    var.shape == [num_test, output_dim]
            If num_inducing_samples is not None,
                means.shape == [num_inducing_samples, num_test, output_dim],
                If full_cov=True and full_output_cov=False,
                    var.shape == [num_inducing_samples, output_dim, num_test, num_test]
                If full_cov=False and full_output_cov=False,
                    var.shape == [num_inducing_samples, num_test, output_dim]
        """
        with tf.name_scope("predict_f") as scope:
            if num_inducing_samples is None:
                # q_mu = self.q_mu
                # q_sqrt = self.q_sqrt
                # mu, var = conditional(Xnew,
                #                       self.inducing_variable,
                #                       self.kernel,
                #                       q_mu,
                #                       q_sqrt=q_sqrt,
                #                       full_cov=full_cov,
                #                       white=self.whiten,
                #                       full_output_cov=full_output_cov)
                return self.posterior(
                    posteriors.PrecomputeCacheType.NOCACHE
                ).fused_predict_f(
                    Xnew, full_cov=full_cov, full_output_cov=full_output_cov
                )
            else:
                q_mu = self.sample_inducing_points(num_inducing_samples)
                q_sqrt = None

                @tf.function
                def single_sample_conditional(q_mu):
                    return conditional(
                        Xnew,
                        self.inducing_variable,
                        self.kernel,
                        q_mu,
                        q_sqrt=q_sqrt,
                        full_cov=full_cov,
                        white=self.whiten,
                        full_output_cov=full_output_cov,
                    )

                mu, var = tf.map_fn(
                    single_sample_conditional,
                    q_mu,
                    fn_output_signature=(default_float(), default_float()),
                )
            return mu + self.mean_function(Xnew), var

    def predict_y(
        self,
        Xnew: InputData,
        num_inducing_samples: int = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """Compute mean and variance at Xnew."""
        f_mean, f_var = self.predict_f(
            Xnew,
            num_inducing_samples=num_inducing_samples,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
        )
        return self.likelihood.predict_mean_and_var(f_mean, f_var)
