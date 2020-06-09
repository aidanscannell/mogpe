import abc
from typing import Optional

import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Module, Parameter
from gpflow.conditionals import conditional, sample_conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.util import inducingpoint_wrapper
# from utils.util import init_variational_parameters
from util import init_variational_parameters
# from utils.util import init_variational_parameters

tfd = tfp.distributions
kl = tfd.kullback_leibler


class SVGP(Module):
    def __init__(
            self,
            kernel,
            likelihood,
            inducing_variable,
            mean_function=None,
            q_mu=None,
            q_sqrt=None,
            q_diag=False,
            # num_data=None,
            num_latent_gps=None,
            whiten=True):
        super().__init__()
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean_function = mean_function
        # self.num_data = num_data
        self.num_latent_gps = num_latent_gps
        self.whiten = whiten
        self.q_mu = q_mu
        self.q_sqrt = q_sqrt
        self.q_diag = q_diag
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)
        gpf.set_trainable(self.inducing_variable, False)
        # q_mu_transpose = tf.transpose(q_mu, [1, 0])
        # self.q_dist = tfp.distributions.MultivariateNormalTriL(
        #     loc=q_mu_transpose,
        #     # loc=self.q_mu,
        #     scale_tril=self.q_sqrt,
        #     validate_args=False,
        #     allow_nan_stats=True,
        #     name='q(U)')

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

        return mu + self.mean_function(Xnew), var

    def predict_f_samples(self,
                          Xnew: InputData,
                          num_samples: Optional[int] = None,
                          full_cov: bool = True,
                          full_output_cov: bool = False) -> tf.Tensor:
        """ Produce samples from the posterior latent function(s) at the input points."""
        if full_cov and full_output_cov:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not supported."
            )
        mean, cov = self.predict_f(Xnew,
                                   full_cov=full_cov,
                                   full_output_cov=full_output_cov)
        return self._sample_mvn(mean, cov, num_samples, full_cov,
                                full_output_cov)

    def _sample_mvn(self,
                    mean,
                    cov,
                    num_samples,
                    full_cov=False,
                    full_output_cov=False):
        if full_cov:
            # mean: [..., N, P]
            # cov: [..., P, N, N]
            mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
            samples = sample_mvn(mean_for_sample,
                                 cov,
                                 "full",
                                 num_samples=num_samples)  # [..., (S), P, N]
            samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        else:
            # mean: [..., N, P]
            # cov: [..., N, P] or [..., N, P, P]
            cov_structure = "full" if full_output_cov else "diag"
            samples = sample_mvn(mean,
                                 cov,
                                 cov_structure,
                                 num_samples=num_samples)  # [..., (S), N, P]
        return samples  # [..., (S), N, P]

    def predict_y(self,
                  Xnew: InputData,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        """ Compute the mean and variance of the held-out data at the input points. """
        f_mean, f_var = self.predict_f(Xnew,
                                       full_cov=full_cov,
                                       full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)


class ExpertsBase(Module):
    def __init__(self,
                 output_dim,
                 kernels,
                 mean_functions,
                 noise_vars,
                 num_samples=None,
                 whiten=True,
                 num_data=None):
        self.output_dim = output_dim
        self.kernels = kernels
        self.mean_functions = mean_functions
        self.noise_vars = noise_vars
        self.num_samples = num_samples
        self.whiten = whiten
        self.num_data = num_data
        self.num_latent_gps = output_dim

    @abc.abstractmethod
    def _init_experts(self, *args, **kwargs):
        raise NotImplementedError

    def _init_expert(self, kernel, mean_function, inducing_variable, q_mu,
                     q_sqrt, q_diag, noise_var):
        likelihood = gpf.likelihoods.Gaussian(noise_var[0])
        # likelihood = GaussianVec(noise_var)

        # TODO make likelihood have separate noise variances for each output dimension
        # This likelihood switches between Gaussian noise with different variances for each f_i:
        # lik_list = []
        # for var in noise_var:
        #     lik_list.append(gpf.likelihoods.Gaussian(var))
        # likelihood = gpf.likelihoods.SwitchedLikelihood(lik_list)

        # from gpflow.utilities import print_summary
        # print_summary(likelihood)

        expert_svgp = SVGP(
            kernel,
            likelihood,
            inducing_variable,
            mean_function,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            q_diag=q_diag,
            # num_data=self.num_data,
            num_latent_gps=self.output_dim)
        return expert_svgp

    def prior_kls(self):
        kls = []
        for expert in self.experts:
            kls.append(expert.prior_kl())
        return kls

    def _sample_expert(self, expert, X, inducing_samples, num_samples=1):
        # return expert.predict_f_samples(X,
        #                                 full_cov=False,
        #                                 full_output_cov=False,
        #                                 num_samples=num_samples)

        f_mean, f_var = expert.predict_f(X, inducing_samples=inducing_samples)
        if expert.num_samples is None:
            print('analytic expert expectation')
            expected_prob_y = tf.exp(
                expert.likelihood.predict_log_density(f_mean, f_var, Y))
            print('expected_prob_y')
            print(expected_prob_y)
        else:
            print('sampling expert expectation')
            f_samples = expert._sample_mvn(f_mean,
                                           f_var,
                                           num_samples,
                                           full_cov=False)
            # f_samples = expert.predict_f_samples(X,
            #                                      num_samples_f,
            #                                      full_cov=False)
            prob_y = tf.exp(expert.likelihood._log_prob(f_samples, Y))
            print('prob_y')
            print(prob_y)
            expected_prob_y = 1. / num_samples_f * tf.reduce_sum(prob_y, 0)
            print('expected_prob_y')
            print(expected_prob_y)
        return expected_prob_y

    # def _expert_expectation(self, expert, X, Y):
    #     if self.num_samples is None:
    #         f_mean, f_var = expert.predict_f(X)
    #         log_prob = expert.likelihood.predict_log_density(f_mean, f_var, Y)
    #     else:
    #         f_samples = self._sample_expert(expert, X, self.num_samples)
    #         log_prob = expert.likelihood._log_prob(f_samples, Y)
    #         # TODO divide by num samples

    def _expert_expectation_sample_inducing(self,
                                            expert,
                                            X,
                                            Y,
                                            num_samples_inducing=None):
        print('expert expectation sample inducing')
        num_inducing = expert.q_sqrt.shape[-1]
        u_samples = expert.sample_inducing_points(num_samples_inducing)
        u_samples = tf.transpose(u_samples, [0, 2, 1])
        # TODO correct this reshape
        u_samples = tf.reshape(u_samples, [num_inducing, 1])

        print('u_samples')
        print(u_samples)
        if self.num_samples_f is None:
            print('analytic expert expectation')
            f_mean, f_var = expert.predict_f(X, inducing_samples=u_samples)
            expected_prob_y = tf.exp(
                expert.likelihood.predict_log_density(f_mean, f_var, Y))
            print('expected_prob_y')
            print(expected_prob_y)
        else:
            print('sampling expert expectation')
            f_samples = expert.predict_f_samples(X,
                                                 num_samples_f,
                                                 full_cov=False)
            prob_y = tf.exp(expert.likelihood._log_prob(f_samples, Y))
            print('prob_y')
            print(prob_y)
            expected_prob_y = 1. / num_samples_f * tf.reduce_sum(prob_y, 0)
            print('expected_prob_y')
            print(expected_prob_y)

        # print('f_samples')
        # print(f_mean)
        # print(f_var)
        # expected_prob_y = tf.exp(
        #     expert.likelihood.predict_log_density(f_mean, f_var, Y))

        return expected_prob_y

    # def _expert_expectation_sample_inducing(self,
    #                                         expert,
    #                                         X,
    #                                         Y,
    #                                         num_samples_inducing=None):
    #     print('expert expectation sample inducing')
    #     num_inducing = expert.q_sqrt.shape[-1]
    #     u_samples = expert.sample_inducing_points(num_samples_inducing)
    #     u_samples = tf.transpose(u_samples, [0, 2, 1])
    #     # TODO correct this reshape
    #     u_samples = tf.reshape(u_samples, [num_inducing, 1])

    #     print('u_samples')
    #     print(u_samples)
    #     if self.num_samples_f is None:
    #         print('analytic expert expectation')
    #         f_mean, f_var = expert.predict_f(X, inducing_samples=u_samples)
    #         expected_prob_y = tf.exp(
    #             expert.likelihood.predict_log_density(f_mean, f_var, Y))
    #         print('expected_prob_y')
    #         print(expected_prob_y)
    #     else:
    #         print('sampling expert expectation')
    #         f_samples = expert.predict_f_samples(X,
    #                                              num_samples_f,
    #                                              full_cov=False)
    #         prob_y = tf.exp(expert.likelihood._log_prob(f_samples, Y))
    #         print('prob_y')
    #         print(prob_y)
    #         expected_prob_y = 1. / num_samples_f * tf.reduce_sum(prob_y, 0)
    #         print('expected_prob_y')
    #         print(expected_prob_y)

    #     # print('f_samples')
    #     # print(f_mean)
    #     # print(f_var)
    #     # expected_prob_y = tf.exp(
    #     #     expert.likelihood.predict_log_density(f_mean, f_var, Y))

    #     return expected_prob_y

    # def experts_expectations_sample_inducing(self,
    #                                          X,
    #                                          Y,
    #                                          num_samples_inducing=None):
    #     expected_experts = []
    #     for expert in self.experts:
    #         expected_expert = self._expert_expectation_sample_inducing(
    #             expert, X, Y, num_samples_inducing)
    #         expected_experts.append(expected_expert)
    #     return expected_experts

    # def _expert_expectation(self, expert, X, Y, num_samples=None):
    #     if num_samples is None:
    #         f_mean, f_var = expert.predict_f(X)
    #         log_prob = expert.likelihood.predict_log_density(f_mean, f_var, Y)
    #     else:
    #         f_samples = self._sample_expert(expert, X, num_samples)
    #         log_prob = expert.likelihood._log_prob(f_samples, Y)
    #         # TODO divide by num samples
    #     print('expert prob')
    #     print(log_prob)
    #     return tf.exp(log_prob)
    # def expert_conditional(self, expert, X):
    #     print('expert expectation sample inducing')
    #     if self.num_samples is None:
    #         f_mean, f_var = expert.predict_f(X)
    #     else:
    #         num_inducing = expert.q_sqrt.shape[-1]
    #         u_samples = expert.sample_inducing_points(num_samples_inducing)
    #         u_samples = tf.transpose(u_samples, [0, 2, 1])
    #         # TODO correct this reshape
    #         u_samples = tf.reshape(u_samples, [num_inducing, 1])

    #     print('u_samples')
    #     print(u_samples)
    #     if self.num_samples_f is None:
    #         print('analytic expert expectation')
    #         f_mean, f_var = expert.predict_f(X, inducing_samples=u_samples)
    #         expected_prob_y = tf.exp(
    #             expert.likelihood.predict_log_density(f_mean, f_var, Y))
    #         print('expected_prob_y')
    #         print(expected_prob_y)
    #     else:
    #         print('sampling expert expectation')
    #         f_samples = expert.predict_f_samples(X,
    #                                              num_samples_f,
    #                                              full_cov=False)
    #         prob_y = tf.exp(expert.likelihood._log_prob(f_samples, Y))
    #         print('prob_y')
    #         print(prob_y)
    #         expected_prob_y = 1. / num_samples_f * tf.reduce_sum(prob_y, 0)
    #         print('expected_prob_y')
    #         print(expected_prob_y)

    def _expert_expectation(self, expert, X, Y, num_samples_inducing=None):
        if num_samples_inducing is None:
            print('not sampling inducing points')
            f_mean, f_var = expert.predict_f(X)
        else:
            print('sampling inducing points')
            num_inducing = expert.q_sqrt.shape[-1]
            inducing_samples = expert.sample_inducing_points(
                num_samples_inducing)
            inducing_samples = tf.transpose(inducing_samples, [0, 2, 1])
            # TODO correct this reshape
            inducing_samples = tf.reshape(inducing_samples, [num_inducing, 1])
            f_mean, f_var = expert.predict_f(X,
                                             inducing_samples=inducing_samples)
        if self.num_samples is None:
            print('analytic expert expectation')
            expected_prob_y = tf.exp(
                expert.likelihood.predict_log_density(f_mean, f_var, Y))
            print('expected_prob_y')
            print(expected_prob_y)
        else:
            print('sampling expert expectation')
            f_samples = expert._sample_mvn(f_mean,
                                           f_var,
                                           self.num_samples,
                                           full_cov=False)
            # f_samples = expert.predict_f_samples(X,
            #                                      num_samples_f,
            #                                      full_cov=False)
            prob_y = tf.exp(expert.likelihood._log_prob(f_samples, Y))
            print('prob_y')
            print(prob_y)
            expected_prob_y = 1. / self.num_samples * tf.reduce_sum(prob_y, 0)
            print('expected_prob_y')
            print(expected_prob_y)
        return expected_prob_y

        # f_samples = self._sample_expert(expert, X, num_samples_inducing)
        # log_prob = expert.likelihood._log_prob(f_samples, Y)
        # TODO divide by num samples
        # print('expert prob')
        # print(log_prob)
        # return tf.exp(log_prob)

    def experts_expectations(self, X, Y, num_samples_inducing=None):
        expected_experts = []
        for expert in self.experts:
            expected_experts.append(
                self._expert_expectation(expert, X, Y, num_samples_inducing))
            # expected_experts.append(expected_expert)
        return expected_experts

    # def experts_expectations(self, X, Y, num_samples=None):
    #     expected_experts = []
    #     for expert in self.experts:
    #         expected_experts.append(
    #             self._expert_expectation(expert, X, Y, num_samples))
    #         # expected_experts.append(expected_expert)
    #     return expected_experts

    # def _sample_expert_expectation(self, expert, X, Y, num_samples=1):
    #     f_samples = self._sample_expert(expert, X, num_samples)
    #     return 1. / num_samples * tf.reduce_sum(
    #         tf.exp(expert.likelihood._log_prob(f_samples, Y)), 0)

    # def _analytic_expert_expectation(self, expert, X, Y):
    #     f_mean, f_var = expert.predict_f(X)
    #     return tf.exp(expert.likelihood.predict_log_density(f_mean, f_var, Y))

    # def experts_expectations(self, X, Y, num_samples=None):
    #     expected_experts = []
    #     for k, expert in enumerate(self.experts):
    #         if num_samples is None:
    #             expected_expert = self._analytic_expert_expectation(
    #                 expert, X, Y)
    #         else:
    #             expected_expert = self._sample_expert_expectation(
    #                 expert, X, Y, num_samples)
    #         expected_experts.append(expected_expert)
    #     return expected_experts

    def predict_fs(self, Xnew, full_cov=False, full_output_cov=False):
        f_mus, f_vars = [], []
        for expert in self.experts:
            f_mu, f_var = expert.predict_f(Xnew,
                                           full_cov=full_cov,
                                           full_output_cov=full_output_cov)
            f_mus.append(f_mu)
            f_vars.append(f_var)
        return f_mus, f_vars

    def predict_ys(self, Xnew, full_cov=False, full_output_cov=False):
        y_mus, y_vars = [], []
        for expert in self.experts:
            y_mu, y_var = expert.predict_y(Xnew,
                                           full_cov=full_cov,
                                           full_output_cov=full_output_cov)
            y_mus.append(y_mu)
            y_vars.append(y_var)
        return y_mus, y_vars


class ExpertsSeparate(ExpertsBase):
    def __init__(self,
                 inducing_variables,
                 output_dim,
                 kernels,
                 mean_functions,
                 noise_vars,
                 num_samples=None,
                 q_diags=[False, False],
                 q_mus=[None, None],
                 q_sqrts=[None, None],
                 whiten=True,
                 num_data=None):
        super().__init__(output_dim, kernels, mean_functions, noise_vars,
                         num_samples)

        q_mus_ = []
        q_sqrts_ = []
        for inducing_variable, q_mu, q_sqrt, q_diag in zip(
                inducing_variables, q_mus, q_sqrts, q_diags):
            # init variational parameters
            inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
                gpf.inducing_variables.InducingPoints(inducing_variable))
            num_inducing = len(inducing_variable)
            q_mu, q_sqrt = init_variational_parameters(num_inducing, q_mu,
                                                       q_sqrt, q_diag,
                                                       self.num_latent_gps)
            q_mus_.append(q_mu)
            q_sqrts_.append(q_sqrt)

        self.experts = self._init_experts(inducing_variables, q_mus_, q_sqrts_,
                                          q_diags)

    def _init_experts(self, inducing_variables, q_mus, q_sqrts, q_diags):
        experts = []
        for kernel, mean_function, inducing_variable, q_mu, q_sqrt, q_diag, noise_var in zip(
                self.kernels, self.mean_functions, inducing_variables, q_mus,
                q_sqrts, q_diags, self.noise_vars):
            experts.append(
                self._init_expert(kernel, mean_function, inducing_variable,
                                  q_mu, q_sqrt, q_diag, noise_var))
        return experts


class ExpertsShared(ExpertsBase):
    def __init__(self,
                 inducing_variable,
                 output_dim,
                 kernels,
                 mean_functions,
                 noise_vars,
                 q_diag=False,
                 q_mu=None,
                 q_sqrt=None,
                 whiten=True,
                 num_data=None):
        super().__init__(output_dim, kernels, mean_functions, noise_vars)

        # init variational parameters
        inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(inducing_variable))
        num_inducing = len(inducing_variable)
        q_mu, q_sqrt = init_variational_parameters(num_inducing, q_mu, q_sqrt,
                                                   q_diag, self.num_latent_gps)

        self.experts = self._init_experts(inducing_variable, q_mu, q_sqrt,
                                          q_diag)

    def _init_experts(self, inducing_variable, q_mu, q_sqrt, q_diag):
        experts = []
        for kernel, mean_function, noise_var in zip(self.kernels,
                                                    self.mean_functions,
                                                    self.noise_vars):
            experts.append(
                self._init_expert(kernel, mean_function, inducing_variable,
                                  q_mu, q_sqrt, q_diag, noise_var))
        return experts


# class Experts(Module):
#     def __init__(self,
#                  inducing_variable,
#                  output_dim,
#                  kernels,
#                  mean_functions,
#                  noise_vars,
#                  q_diag=False,
#                  q_mu=None,
#                  q_sqrt=None,
#                  whiten=True,
#                  num_data=None):
#         super().__init__()
#         # self.inducing_variable = inducingpoint_wrapper(inducing_variable)
#         # self.inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
#         #     gpf.inducing_variables.InducingPoints(inducing_variable))
#         self.output_dim = output_dim
#         self.q_diag = q_diag
#         self.whiten = whiten
#         self.num_data = num_data
#         self.num_latent_gps = output_dim

#         # init variational parameters
#         self.inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
#             gpf.inducing_variables.InducingPoints(inducing_variable))
#         num_inducing = len(self.inducing_variable)
#         self.q_mu, self.q_sqrt = init_variational_parameters(
#             num_inducing, q_mu, q_sqrt, q_diag, self.num_latent_gps)

#         self.experts = self._init_experts(kernels, mean_functions, noise_vars,
#                                           noise_vars_trainable)

#     def _init_experts(self, kernels, mean_functions, noise_vars,
#                       noise_vars_trainable):
#         experts = []
#         for kernel, mean_function, noise_var, noise_var_trainable in zip(
#                 kernels, mean_functions, noise_vars, noise_vars_trainable):
#             experts.append(
#                 self._init_expert(kernel, mean_function, noise_var,
#                                   noise_var_trainable))
#         return experts

#     def _init_expert(self,
#                      kernel,
#                      mean_function,
#                      noise_var,
#                      noise_var_trainable=True):
#         likelihood = gpf.likelihoods.Gaussian(noise_var[0])
#         # likelihood = GaussianVec(noise_var)

#         # TODO make likelihood have separate noise variances for each output dimension
#         # This likelihood switches between Gaussian noise with different variances for each f_i:
#         # lik_list = []
#         # for var in noise_var:
#         #     lik_list.append(gpf.likelihoods.Gaussian(var))
#         # likelihood = gpf.likelihoods.SwitchedLikelihood(lik_list)

#         # from gpflow.utilities import print_summary
#         # print_summary(likelihood)

#         if noise_var_trainable is False:
#             gpf.set_trainable(likelihood, False)

#         expert_svgp = SVGP(
#             kernel,
#             likelihood,
#             self.inducing_variable,
#             mean_function,
#             self.q_mu,
#             self.q_sqrt,
#             # num_data=self.num_data,
#             num_latent_gps=self.output_dim)
#         return expert_svgp

#     def prior_kls(self):
#         kls = []
#         for expert in self.experts:
#             kls.append(expert.prior_kl())
#         return kls

#     def prior_kl(self, prob_a_0):
#         prob_a_1 = 1 - prob_a_0
#         for expert in self.experts:
#             # print('Kzz')
#             # print(expert.inducing_variable)
#             # print(expert.inducing_variable.inducing_variables)
#             # print(expert.inducing_variable.inducing_variables[0].Z)
#             # print(expert.kernel)
#             Kzz = expert.kernel.K(
#                 expert.inducing_variable.inducing_variables[0].Z)
#             print(Kzz)

#         print('hafasfasf')
#         q_sqrt_1 = self.experts[0].q_sqrt
#         q_sqrt_2 = self.experts[1].q_sqrt
#         q_sqrt_1 = tf.linalg.diag_part(tf.squeeze(q_sqrt_1))
#         q_sqrt_2 = tf.linalg.diag_part(tf.squeeze(q_sqrt_2))
#         q_mu_1 = tf.squeeze(self.experts[0].q_mu)
#         q_mu_2 = tf.squeeze(self.experts[1].q_mu)
#         q_mu = tf.stack([q_mu_1, q_mu_2], axis=0)
#         q_sqrt = tf.stack([q_sqrt_1, q_sqrt_2], axis=0)
#         q_mu = tf.transpose(q_mu, [1, 0])
#         q_sqrt = tf.transpose(q_sqrt, [1, 0])
#         print(q_mu)
#         print(q_sqrt)
#         print(prob_a_0)

#         q_mixture = tfd.MixtureSameFamily(
#             mixture_distribution=tfd.Categorical(probs=prob_a_0),
#             components_distribution=tfd.Normal(loc=q_mu, scale=q_sqrt))
#         print(q_mixture)
#         # p_mixture = tfd.MixtureSameFamily(
#         #     mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
#         #     components_distribution=tfd.Normal(loc=locs, scale=[0.1, 0.5]))

#         # q_mixture = tfd.MixtureSameFamily(
#         #     mixture_distribution=tfd.Categorical(probs=[0.1, 0.9]),
#         #     components_distribution=tfd.Normal(loc=[-1., 1],
#         # scale=[0.1, 0.5]))

#     def _sample_expert(self, expert, X, num_samples=1):
#         return expert.predict_f_samples(X,
#                                         full_cov=False,
#                                         full_output_cov=False,
#                                         num_samples=num_samples)

#     def _sample_expert_expectation(self, expert, X, Y, num_samples=1):
#         f_samples = self._sample_expert(expert, X, num_samples)
#         return 1. / num_samples * tf.reduce_sum(
#             tf.exp(expert.likelihood._log_prob(f_samples, Y)), 0)

#     def _analytic_expert_expectation(self, expert, X, Y):
#         f_mean, f_var = expert.predict_f(X)
#         return tf.exp(expert.likelihood.predict_log_density(f_mean, f_var, Y))

#     def experts_expectations(self, X, Y, num_samples=None):
#         expected_experts = []
#         for k, expert in enumerate(self.experts):
#             if num_samples is None:
#                 expected_expert = self._analytic_expert_expectation(
#                     expert, X, Y)
#             else:
#                 expected_expert = self._sample_expert_expectation(
#                     expert, X, Y, num_samples)
#             expected_experts.append(expected_expert)
#         return expected_experts

#     def predict_fs(self, Xnew, full_cov=False, full_output_cov=False):
#         q_mu = self.q_mu
#         q_sqrt = self.q_sqrt
#         f_mus, f_vars = [], []
#         for expert in self.experts:
#             f_mu, f_var = expert.predict_f(Xnew, full_cov, full_output_cov)
#             f_mus.append(f_mu)
#             f_vars.append(f_var)
#         return f_mus, f_vars

#     def predict_ys(self, Xnew, full_cov=False, full_output_cov=False):
#         y_mus, y_vars = [], []
#         for expert in self.experts:
#             y_mu, y_var = expert.predict_y(Xnew, full_cov, full_output_cov)
#             y_mus.append(y_mu)
#             y_vars.append(y_var)
#         return y_mus, y_vars
