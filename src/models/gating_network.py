import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Module
from gpflow.conditionals import conditional, sample_conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.util import inducingpoint_wrapper
from util import init_variational_parameters


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 -
                                                          2 * jitter) + jitter


class GatingNetwork(Module):
    def __init__(self,
                 kernel,
                 inducing_variable,
                 mean_function,
                 num_latent_gps=1,
                 q_diag=False,
                 q_mu=None,
                 q_sqrt=None,
                 whiten=True,
                 num_data=None):
        self.kernel = kernel
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)
        gpf.set_trainable(self.inducing_variable, False)
        self.mean_function = mean_function
        self.num_latent_gps = num_latent_gps
        self.q_diag = q_diag
        self.whiten = whiten
        self.num_data = num_data

        # init variational parameters
        num_inducing = len(self.inducing_variable)
        self.q_mu, self.q_sqrt = init_variational_parameters(
            num_inducing, q_mu, q_sqrt, q_diag, self.num_latent_gps)

    def prior_kl(self):
        return gpf.kullback_leiblers.prior_kl(self.inducing_variable,
                                              self.kernel,
                                              self.q_mu,
                                              self.q_sqrt,
                                              whiten=self.whiten)

    def sample_inducing_points(self, num_samples=None):
        mu = tf.transpose(self.q_mu, [1, 0])
        q_dist = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            # loc=self.q_mu,
            scale_tril=self.q_sqrt,
            validate_args=False,
            allow_nan_stats=True,
            name='MultivariateNormalQ')
        return q_dist.sample(num_samples)

    def predict_f(self,
                  Xnew: InputData,
                  inducing_samples=None,
                  full_cov=False,
                  full_output_cov=False) -> MeanAndVariance:
        if inducing_samples is None:
            q_mu = self.q_mu
            q_sqrt = self.q_sqrt
        else:
            q_mu = inducing_samples
            q_sqrt = None
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var

    def predict_prob_a_0_given_h(self, h_mean, h_var):
        return 1 - inv_probit(h_mean / (tf.sqrt(1 + h_var)))

    def predict_prob_a_0(self, Xnew: InputData):
        h_mean, h_var = self.predict_f(Xnew)
        return self.predict_prob_a_0_given_h(h_mean, h_var)

    # def predict_prob_a_0_sample_inducing(self, Xnew: InputData, u_samples):
    #     f_mean, f_var = self.predict_f_given_u(Xnew, u_samples)
    #     print('inside predict prob_a_0 sample u')
    #     print(f_mean)
    #     print(f_var)
    #     # f_mean, f_var = self.predict_f(Xnew)
    #     # return 1 - inv_probit(f_samples)
    #     return 1 - inv_probit(f_mean / (tf.sqrt(1 + f_var)))

    def predict_mixing_probs(self, Xnew: InputData):
        prob_a_0 = self.predict_prob_a_0(Xnew)
        prob_a_0 = tf.reshape(prob_a_0, [-1])
        prob_a_1 = 1 - prob_a_0
        return [prob_a_0, prob_a_1]

    # def predict_mixing_probs_sample_u_old(self, Xnew: InputData, num_samples):
    #     num_data = Xnew.shape[0]
    #     print('gating LL^2')
    #     # print(self.q_sqrt)
    #     cov = tf.linalg.diag_part(self.q_sqrt)
    #     # print(cov)
    #     cov = tf.transpose(cov, [1, 0])
    #     # print(cov)
    #     cov = tf.math.square(cov)
    #     # print(cov)
    #     u_samples = sample_mvn(self.q_mu, cov, "diag",
    #                            num_samples=num_samples)  # [..., (S), P, N]
    #     # TODO correct this reshape
    #     u_samples = tf.reshape(u_samples, [num_data, 1])
    #     prob_a_0 = self.predict_prob_a_0_sample_u(Xnew, u_samples)
    #     print('inside predict mixing provs')
    #     print(prob_a_0)
    #     prob_a_0 = tf.reshape(prob_a_0, [-1])
    #     print(prob_a_0)
    #     prob_a_1 = 1 - prob_a_0
    #     return [prob_a_0, prob_a_1]

    def predict_mixing_probs_sample_inducing(self,
                                             Xnew: InputData,
                                             num_samples_inducing=None):
        print('gating expectation sample inducing')
        u_samples = self.sample_inducing_points(num_samples_inducing)
        # q_mu = tf.transpose(self.q_mu, [1, 0])
        # q_sqrt = self.q_sqrt
        # print(q_mu)
        # print(q_sqrt)
        # num_sample = None
        # # u_samples = sample_mvn(mu, cov, "full",
        # #                        num_samples=num_samples)  # [..., (S), P, N]
        # q = tfp.distributions.MultivariateNormalTriL(
        #     loc=q_mu,
        #     scale_tril=q_sqrt,
        #     validate_args=False,
        #     allow_nan_stats=True,
        #     name='MultivariateNormalFullCovariance')
        # print('tf mvn')
        # print(q)
        # u_samples = q.sample(num_samples)
        print('u_samples')
        print(u_samples)
        u_samples = tf.transpose(u_samples, [0, 2, 1])
        print(u_samples)
        # TODO correct this reshape
        u_samples = tf.reshape(u_samples, [-1, 1])
        # u_samples = tf.reshape(u_samples, [1, -1])
        print('aaaa')
        print(u_samples)

        print('analytic gating expectation')
        h_mean, h_var = self.predict_f(Xnew, inducing_samples=u_samples)
        print('h_mean and h_var')
        print(h_mean)
        print(h_var)
        expected_prob_a_0 = self.predict_prob_a_0_given_h(h_mean, h_var)
        print('expected_prob_a_0')
        print(expected_prob_a_0)
        # h_samples, h_mean, h_var = sample_conditional(
        #     Xnew,
        #     self.inducing_variable,
        #     self.kernel,
        #     u_samples,
        #     full_cov=False,
        #     full_output_cov=False,
        #     q_sqrt=None,
        #     white=self.whiten,
        #     num_samples=num_samples,
        # )
        # print('h_samples')
        # print(h_samples)

        return [expected_prob_a_0, 1 - expected_prob_a_0]

    def predict_mixing_probs_tensor(self, Xnew: InputData):
        return tf.stack(self.predict_mixing_probs(Xnew))

    # def predict_f_given_u(self,
    #                       Xnew: InputData,
    #                       u_sample,
    #                       full_cov=False,
    #                       full_output_cov=False) -> MeanAndVariance:
    #     print('inside predict f given u')
    #     print(Xnew)
    #     print(u_sample)
    #     mu, var = conditional(
    #         Xnew,
    #         self.inducing_variable,
    #         self.kernel,
    #         u_sample,
    #         q_sqrt=None,
    #         full_cov=full_cov,
    #         white=self.whiten,
    #         full_output_cov=full_output_cov,
    #     )
    #     return mu + self.mean_function(Xnew), var
