#!/usr/bin/env python3
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.conditionals import conditional
from gpflow.config import default_float
from gpflow.models import SVGP
from mogpe.custom_types import InputData

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
