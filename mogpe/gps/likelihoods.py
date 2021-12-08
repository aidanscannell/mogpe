#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from gpflow import logdensities
from gpflow.base import Parameter
from gpflow.likelihoods.base import ScalarLikelihood
from gpflow.utilities import positive

# from gpflow.likelihoods.utils import inv_probit


class Gaussian(ScalarLikelihood):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.
    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(
        self, variance=1.0, variance_lower_bound=DEFAULT_VARIANCE_LOWER_BOUND, **kwargs
    ):
        """
        :param variance: The noise variance; must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        # if variance <= variance_lower_bound:
        #     raise ValueError(
        #         f"The variance of the Gaussian likelihood must be strictly greater than {variance_lower_bound}"
        #     )

        self.variance = Parameter(
            variance, transform=positive(lower=variance_lower_bound)
        )

    def _scalar_log_prob(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def _predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def _predict_log_density(self, Fmu, Fvar, Y):
        return tf.reduce_sum(
            logdensities.gaussian(Y, Fmu, Fvar + self.variance), axis=-1
        )

    def _variational_expectations(self, Fmu, Fvar, Y):
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance,
            axis=-1,
        )
