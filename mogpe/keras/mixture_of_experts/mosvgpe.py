#!/usr/bin/env python3
from typing import List, Optional

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from mogpe.custom_types import Dataset, InputData, DatasetBatch

from ..experts import SVGPExpert
from ..gating_networks import SVGPGatingNetwork
from .base import MixtureOfExpertsBase

tf.keras.backend.set_floatx("float64")
tfd = tfp.distributions
tfpl = tfp.layers


class MixtureOfSVGPExperts(MixtureOfExpertsBase):
    """Mixture of SVGP experts using stochastic variational inference.

    Implemention of a Mixture of Gaussian Process Experts method where both
    the gating network and experts are modelled as SVGPs.
    The model is trained with stochastic variational inference by exploiting
    the factorization achieved by sparse GPs.
    """

    def __init__(
        self,
        experts_list: List[SVGPExpert],
        gating_network: SVGPGatingNetwork,
        num_samples: Optional[int] = 1,
        num_data: Optional[int] = None,
        bound: Optional[str] = "further_gating",  # "further_gating"/"tight"/"further"
        name: str = "MoSVGPE",
    ):
        for expert in experts_list:
            assert isinstance(expert, SVGPExpert)
        assert isinstance(gating_network, SVGPGatingNetwork)
        super().__init__(
            experts_list=experts_list, gating_network=gating_network, name=name
        )
        self._num_data = num_data
        self._num_samples = num_samples
        self._bound = bound
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # self.gating_kl_tracker = tf.keras.metrics.Mean(name="gating_kl")
        # self.experts_kl_tracker = tf.keras.metrics.Mean(name="expert_kl")

    def call(
        self, Xnew: InputData, training: Optional[bool] = False
    ) -> tfd.MixtureSameFamily:
        if not training:
            # return self.predict_y(Xnew)
            dist = self.predict_y(Xnew)
            return dist.mean(), dist.variance()

    def train_step(self, data: DatasetBatch):
        with tf.GradientTape() as tape:
            loss = -self.maximum_log_likelihood_objective(data)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
            # "gating_kl": self.gating_kl_tracker.result(),
            # "experts_kl": self.experts_kl_tracker.result(),
        }

    def test_step(self, data):
        loss = -self.maximum_log_likelihood_objective(data)
        return {"loss": loss}

    def maximum_log_likelihood_objective(self, data: Dataset) -> ttf.Tensor0:
        return self.elbo(
            data=data,
            num_samples=self.num_samples,
            num_data=self.num_data,
            bound=self.bound,
        )

    def elbo(
        self,
        data: Dataset,
        num_samples: Optional[int] = 1,
        num_data: Optional[int] = None,
        bound: Optional[str] = "further_gating",  # "further_gating"/"tight"/"further"
    ) -> ttf.Tensor0:
        """MoSVGPE Evidence Lower BOund (ELBO)"""
        if bound == "further_gating":
            return self.lower_bound_further_gating(
                data, num_data=num_data, num_samples=num_samples
            )
        elif bound == "tight":
            return self.lower_bound_tight(
                data, num_data=num_data, num_samples=num_samples
            )
        elif bound == "further":
            return self.lower_bound_further(
                data, num_data=num_data, num_samples=num_samples
            )
        else:
            print(
                'bound should be "tight" or "further_gating" or "further". Using further_gating as default.'
            )
            return self.lower_bound_further_gating(data, num_samples)

    def lower_bound_further_gating(
        self,
        data_batch: DatasetBatch,
        num_data: Optional[int] = 1,
        num_samples: Optional[int] = 1,
    ) -> ttf.Tensor0:
        r"""Lower bound to the log-marginal likelihood (ELBO).

        Similar to lower_bound_tight but with a further bound on the gating
        network. The bound removes the M dimensional integral over the gating
        network inducing variables $q(\hat{\mathbf{U}})$ with 1 dimensional
        integrals over the gating network variational posterior $q(\mathbf{h}_n)$.
        """
        X, Y = data_batch

        kl_gating = tf.reduce_sum(self.gating_network.prior_kl())
        kl_experts = tf.reduce_sum([expert.prior_kl() for expert in self.experts_list])
        # self.gating_kl_tracker.update_state(kl_gating)
        # self.experts_kl_tracker.update_state(kl_experts)

        # Evaluate gating network to get samples of categorical dist over inicator var
        # mixing_probs = self.gating_network(X, num_samples, training=True)  # [S, N, K]
        mixing_probs = self.gating_network.predict_categorical_dist_given_h_samples(
            X, num_h_samples=num_samples
        ).probs  # [S, N, K]
        print("Mixing probs: {}".format(mixing_probs.shape))

        # Evaluate experts
        Y = tf.expand_dims(Y, 0)  # [S, N, F]
        # print("Y: {}".format(Y.shape))
        experts_probs = [
            expert.predict_dist_given_inducing_samples(X, num_samples).prob(Y)
            for expert in self.experts_list
        ]
        experts_probs = tf.stack(experts_probs, -1)  # [S, N, K]
        print("Experts probs: {}".format(experts_probs.shape))

        tf.debugging.assert_shapes(
            [
                (experts_probs, ["S1", "N", "K"]),
                (mixing_probs, ["S2", "N", "K"]),
            ],
            message="Gating network and experts dimensions do not match",
        )

        # Expand to enable integrationg over both expert and gating samples
        experts_probs = experts_probs[:, tf.newaxis, :, tf.newaxis, :]
        mixing_probs = mixing_probs[tf.newaxis, :, :, :, tf.newaxis]
        # print("Experts probs EXP: {}".format(experts_probs.shape))
        # print("Mixing probs EXP: {}".format(mixing_probs.shape))
        # print("Matmul EXP: {}".format(tf.matmul(experts_probs, mixing_probs).shape))
        marginalised_experts = tf.matmul(experts_probs, mixing_probs)[..., 0, 0]
        # print("Marginalised indicator variable: {}".format(marginalised_experts.shape))

        log_prob = tf.math.log(marginalised_experts)
        var_exp = tf.reduce_mean(log_prob, axis=[0, 1])  # Average gating/expert samples
        var_exp = tf.reduce_sum(var_exp, 0)

        scale = variational_expectation_scale(self.num_data, batch_size=tf.shape(X)[0])
        return var_exp * scale - kl_gating - kl_experts

    def lower_bound_tight(
        self,
        data_batch: DatasetBatch,
        num_data: Optional[int] = 1,
        num_samples: Optional[int] = 1,
    ) -> ttf.Tensor0:
        r"""Lower bound to the log-marginal likelihood (ELBO).

        Tighter bound than lower_bound_further but requires an M dimensional
        expectation over the inducing variables $q(\hat{f}, \hat{h})$
        to be approximated (with Gibbs sampling).
        """
        X, Y = data_batch

        kl_gating = tf.reduce_sum(self.gating_network.prior_kl())
        kl_experts = tf.reduce_sum([expert.prior_kl() for expert in self.experts_list])

        # Evaluate gating network to get samples of categorical dist over inicator var
        mixing_probs = (
            self.gating_network.predict_categorical_dist_given_inducing_samples(
                X, num_inducing_samples=num_samples
            ).probs
        )
        print("Mixing probs: {}".format(mixing_probs.shape))

        # Evaluate experts
        Y = tf.expand_dims(Y, 0)  # [S, N, F]
        experts_probs = [
            expert.predict_dist_given_inducing_samples(X, num_samples).prob(Y)
            for expert in self.experts_list
        ]
        experts_probs = tf.stack(experts_probs, -1)  # [S, N, K]
        print("Experts probs: {}".format(experts_probs.shape))

        tf.debugging.assert_shapes(
            [
                (experts_probs, ["S1", "N", "K"]),
                (mixing_probs, ["S2", "N", "K"]),
            ],
            message="Gating network and experts dimensions do not match",
        )
        # Expand to enable integrationg over both expert and gating samples
        experts_probs = experts_probs[:, tf.newaxis, :, tf.newaxis, :]
        mixing_probs = mixing_probs[tf.newaxis, :, :, :, tf.newaxis]
        marginalised_experts = tf.matmul(experts_probs, mixing_probs)[..., 0, 0]
        print("Marginalised indicator variable: {}".format(marginalised_experts.shape))

        log_prob = tf.math.log(marginalised_experts)
        var_exp = tf.reduce_mean(log_prob, axis=[0, 1])  # Average gating/expert samples
        var_exp = tf.reduce_sum(var_exp, 0)
        scale = variational_expectation_scale(num_data, batch_size=tf.shape(X)[0])
        return var_exp * scale - kl_gating - kl_experts

    def lower_bound_further(
        self,
        data_batch: DatasetBatch,
        num_data: Optional[int] = 1,
        num_samples: Optional[int] = 1,
    ) -> ttf.Tensor0:
        r"""Lower bound to the log-marginal likelihood (ELBO).

        Looser bound than lower_bound_tight as it marginalises both of the expert's
        and the gating network's inducing variables $q(\hat{f}, \hat{h})$ in closed-form.
        Replaces M-dimensional approx integrals with 1-dimensional approx integrals.

        This bound is equivalent to a different likelihood approximation that
        only mixes the noise models (as opposed to the full GPs).
        """
        X, Y = data_batch

        kl_gating = tf.reduce_sum(self.gating_network.prior_kl())
        kl_experts = tf.reduce_sum([expert.prior_kl() for expert in self.experts_list])

        # Evaluate gating network to get samples of categorical dist over inicator var
        # mixing_probs = self.gating_network(X, num_samples, training=True)  # [S, N, K]
        mixing_probs = self.gating_network.predict_categorical_dist_given_h_samples(
            X, num_h_samples=num_samples
        ).probs  # [S, N, K]

        # Sample each experts variational posterior q(F) and construct p(Y|F)
        Y = tf.expand_dims(Y, 0)  # [S, N, F]
        experts_probs = [
            expert.predict_dist_given_f_samples(X, num_f_samples=num_samples).prob(Y)
            for expert in self.experts_list
        ]
        experts_probs = tf.stack(experts_probs, -1)  # [S, N, K]

        tf.debugging.assert_shapes(
            [
                (experts_probs, ["S1", "N", "K"]),
                (mixing_probs, ["S2", "N", "K"]),
            ],
            message="Gating network and experts dimensions do not match",
        )
        # Expand to enable integrationg over both expert and gating samples
        experts_probs = experts_probs[:, tf.newaxis, :, tf.newaxis, :]
        mixing_probs = mixing_probs[tf.newaxis, :, :, :, tf.newaxis]
        marginalised_experts = tf.matmul(experts_probs, mixing_probs)[..., 0, 0]
        print("Marginalised indicator variable: {}".format(marginalised_experts.shape))

        log_prob = tf.math.log(marginalised_experts)
        var_exp = tf.reduce_mean(log_prob, axis=[0, 1])  # Average gating/expert samples
        var_exp = tf.reduce_sum(var_exp, 0)
        scale = variational_expectation_scale(num_data, batch_size=tf.shape(X)[0])
        return var_exp * scale - kl_gating - kl_experts

    @property
    def metrics(self):
        return [self.loss_tracker]

    @property
    def num_data(self):
        return self._num_data

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def bound(self):
        return self._bound


def variational_expectation_scale(num_data, batch_size):
    if num_data is not None:
        return num_data / batch_size
        # batch_size = tf.cast(tf.shape(X)[0], default_float())
        # scale = tf.cast(num_data, default_float()) / batch_size
    else:
        return 1.0
        # scale = tf.cast(1.0, default_float())
