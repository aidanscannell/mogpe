from typing import Tuple

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from experts import ExpertsSeparate, ExpertsShared
from gating_network import GatingNetwork
from gpflow.base import Parameter
from gpflow.ci_utils import ci_niter
from gpflow.config import default_float
from gpflow.models import BayesianModel, ExternalDataTrainingLossMixin
from gpflow.models.model import InputData, MeanAndVariance, RegressionData
from gpflow.models.util import inducingpoint_wrapper
from gpflow.utilities import positive, print_summary, triangular
# from plot_model import plot_and_save
# from util import init_experts, init_inducing_variables, run_adam
from util import init_inducing_variables, run_adam
# from util import init_experts, init_inducing_variables, run_adam

tfd = tfp.distributions


class SVMoGPE(BayesianModel, ExternalDataTrainingLossMixin):
    """ Stochastic Variational Mixtures of 2 Gaussian Process Experts Class. """
    def __init__(self,
                 input_dim,
                 output_dim,
                 experts,
                 gating_network,
                 bound='tight',
                 num_data=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bound = bound
        self.num_data = num_data

        self.experts = experts
        self.gating_network = gating_network

    def build_prior_KL(self, feature, kern, q_mu, q_sqrt):
        if self.whiten:
            K = None
        else:
            K = Kuu(feature, kern,
                    jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return gpf.kullback_leiblers.gauss_kl(q_mu, q_sqrt, K)

    def lower_bound_1(self, X, Y, kl_gating, kl_experts):

        num_samples_inducing = 1

        mixing_probs = self.gating_network.predict_mixing_probs_sample_inducing(
            X, num_samples_inducing)

        # expected_experts = self.experts.experts_expectations_sample_inducing(
        #     X, Y, num_samples_inducing)
        expected_experts = self.experts.experts_expectations(
            X, Y, num_samples_inducing)

        sum_over_indicator = 0
        for expected_expert, mixing_prob in zip(expected_experts,
                                                mixing_probs):
            print('inside loop')
            print(expected_expert)
            print(mixing_prob)
            mixing_prob = tf.reshape(mixing_prob, [-1])
            print(mixing_prob)
            sum_over_indicator += expected_expert * mixing_prob

        # TODO divide by num inducing point samples
        var_exp = tf.reduce_sum(tf.math.log(sum_over_indicator))
        print('var_exp')
        print(var_exp)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl_gating.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl_gating.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_gating.dtype)

        return tf.reduce_sum(var_exp) * scale - kl_gating - kl_experts

    def lower_bound_2(self, X, Y, kl_gating, kl_experts):

        mixing_probs = self.gating_network.predict_mixing_probs(X)
        # expected_experts = self.experts.experts_expectations(X, Y)
        expected_experts = self.experts.experts_expectations(
            X, Y, num_samples_inducing=None)

        sum_over_indicator = 0
        for expected_expert, mixing_prob in zip(expected_experts,
                                                mixing_probs):
            print('inside loop')
            print(expected_expert)
            print(mixing_prob)
            sum_over_indicator += expected_expert * mixing_prob

        var_exp = tf.reduce_sum(tf.math.log(sum_over_indicator))
        # var_exp = self._var_expectation(X, Y, num_samples)
        print('var_exp')
        print(var_exp)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl_gating.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl_gating.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_gating.dtype)

        # return tf.reduce_sum(var_exp) * scale
        return tf.reduce_sum(var_exp) * scale - kl_gating - kl_experts

    def lower_bound_3(self, X, Y, kl_gating, kl_experts):
        ''' bound on joint prob p(y, alpha | x) '''
        num_samples_f = 1
        # num_samples_f = None

        # mixing_probs = self.gating_network.predict_mixing_probs(X)
        expected_experts = self.experts.experts_expectations(
            X, Y, num_samples_f)
        print('expected_experts')
        print(expected_experts)
        var_exp = tf.reduce_sum(tf.math.log(expected_experts))
        print(var_exp)

        # sum_over_indicator = 0
        # for expected_expert, mixing_prob in zip(expected_experts,
        #                                         mixing_probs):
        #     sum_over_indicator += expected_expert * mixing_prob

        # var_exp = tf.reduce_sum(tf.math.log(sum_over_indicator))
        # var_exp = self._var_expectation(X, Y, num_samples)
        print('var_exp')
        print(var_exp)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl_gating.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl_gating.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_gating.dtype)

        return tf.reduce_sum(var_exp) * scale - kl_gating - kl_experts

    def maximum_log_likelihood_objective(self, data: Tuple[tf.Tensor,
                                                           tf.Tensor]):
        X, Y = data
        kl_gating = self.gating_network.prior_kl()
        kls_experts = self.experts.prior_kls()
        kl_experts = tf.reduce_sum(kls_experts)
        if self.bound == 'tight':
            print('using tight bound')
            return self.lower_bound_1(X, Y, kl_gating, kl_experts)
        elif self.bound == 'titsias':
            print('using titsias bound')
            return self.lower_bound_2(X, Y, kl_gating, kl_experts)
        else:
            error_str = "No bound corresponding to " + str(
                self.bound
            ) + " has been implemented. Select either \'tight\' or \'further\'."
            NotImplementedError(error_str)
        # return self.lower_bound_3(X, Y, kl_gating, kl_experts)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This returns the evidence lower bound (ELBO) of the log marginal likelihood.
        """
        return self.maximum_log_likelihood_objective(data)

    def predict_mixing_probs(self, Xnew: InputData):
        """
        Compute the predictive mixing probabilities [P(a=k | Xnew, ...)]^K
        """
        return self.gating_network.predict_mixing_probs(Xnew)

    def predict_gating_h(self,
                         Xnew: InputData,
                         full_cov: bool = False,
                         full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and (co)variance of the gating network GP at the input points Xnew.
        """
        mean_gating, var_gating = self.gating_network.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return mean_gating, var_gating

    def predict_experts_fs(self,
                           Xnew: InputData,
                           full_cov: bool = False,
                           full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and (co)variance of the GPs
        associated with the experts at the input points Xnew.
        """
        f_means, f_vars = self.experts.predict_fs(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return f_means, f_vars

    def predict_experts_ys(self,
                           Xnew: InputData,
                           full_cov: bool = False,
                           full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and (co)variance of the experts (GP+likelihood) at the input points Xnew.
        """
        y_means, y_vars = self.experts.predict_ys(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return y_means, y_vars

    def predict_y_moment_matched(
            self,
            Xnew: InputData,
            full_cov: bool = False,
            full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the moment matched mean and covariance of the held-out data at the input points.
        TODO is multivariate moment matching implemented correctly. (I used univariate var equation)
        """
        y_means, y_vars = self.predict_experts_ys(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        # mixing_probs = self.predict_mixing_probs(Xnew)
        y_means = tf.convert_to_tensor(y_means)
        y_vars = tf.convert_to_tensor(y_vars)

        mixing_probs = self.gating_network.predict_mixing_probs_tensor(Xnew)
        mixing_probs = tf.expand_dims(mixing_probs, -1)

        # move mixture dimension to last dimension
        y_means = tf.transpose(y_means, [1, 2, 0])
        y_vars = tf.transpose(y_vars, [1, 2, 0])
        mixing_probs = tf.transpose(mixing_probs, [1, 2, 0])

        gaussian_mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixing_probs),
            components_distribution=tfd.Normal(
                loc=y_means,  # One for each component.
                scale=y_vars))  # And same here.

        y_mean = gaussian_mixture.mean()
        y_var = gaussian_mixture.variance()

        return y_mean, y_var

    def sample_y(self,
                 Xnew: InputData,
                 num_samples=100,
                 full_cov: bool = False,
                 full_output_cov: bool = False) -> MeanAndVariance:
        y_means, y_vars = self.predict_experts_ys(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        # mixing_probs = self.predict_mixing_probs(Xnew)
        y_means = tf.convert_to_tensor(y_means)
        y_vars = tf.convert_to_tensor(y_vars)

        mixing_probs = self.gating_network.predict_mixing_probs_tensor(Xnew)
        mixing_probs = tf.expand_dims(mixing_probs, -1)

        # move mixture dimension to last dimension
        y_means = tf.transpose(y_means, [1, 2, 0])
        y_vars = tf.transpose(y_vars, [1, 2, 0])
        mixing_probs = tf.transpose(mixing_probs, [1, 2, 0])

        gaussian_mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixing_probs),
            components_distribution=tfd.Normal(
                loc=y_means,  # One for each component.
                scale=y_vars))  # And same here.

        return gaussian_mixture.sample(num_samples)


def trim_dataset(X, Y, x1_low=-3., x2_low=-3., x1_high=0., x2_high=-1.):
    mask_0 = X[:, 0] < x1_low
    mask_1 = X[:, 1] < x2_low
    mask_2 = X[:, 0] > x1_high
    mask_3 = X[:, 1] > x2_high
    mask = mask_0 | mask_1 | mask_2 | mask_3
    X_partial = X[mask, :]
    Y_partial = Y[mask, :]
    x1 = [x1_low, x1_low, x1_high, x1_high, x1_low]
    x2 = [x2_low, x2_high, x2_high, x2_low, x2_low]
    X_missing = [x1, x2]

    print("New data shape:", Y_partial.shape)
    return X_partial, Y_partial


# if __name__ == "__main__":
#     # import data
#     data = np.load('../data/npz/turbulence/model_data_fan_fixed_subset.npz')
#     X = data['x']
#     Y = data['y'][:, 0:1]
#     # remove some data points
#     X_subset, Y_subset = trim_dataset(X,
#                                       Y,
#                                       x1_low=-3.,
#                                       x2_low=-3.,
#                                       x1_high=0.,
#                                       x2_high=-1.)
#     X = tf.convert_to_tensor(X_subset, dtype=default_float())
#     Y = tf.convert_to_tensor(Y_subset, dtype=default_float())
#     print("Input data shape: ", X.shape)
#     print("Output data shape: ", Y.shape)
#     # standardise input
#     mean_x, var_x = tf.nn.moments(X, axes=[0])
#     mean_y, var_y = tf.nn.moments(Y, axes=[0])
#     X = (X - mean_x) / tf.sqrt(var_x)
#     Y = (Y - mean_y) / tf.sqrt(var_y)
#     data = (X, Y)

#     # data = np.load('../data/artificial/artificial-2d-mixture.npz')

#     # X = tf.convert_to_tensor(data['x'], dtype=default_float())
#     # Y = tf.convert_to_tensor(data['y'], dtype=default_float())
#     # Y = tf.reshape(Y, [-1, 1])
#     # print("Input data shape: ", X.shape)
#     # print("Output data shape: ", Y.shape)
#     # # standardise input
#     # # mean_x, var_x = tf.nn.moments(X, axes=[0])
#     # # mean_y, var_y = tf.nn.moments(Y, axes=[0])
#     # # X = (X - mean_x) / tf.sqrt(var_x)
#     # # Y = (Y - mean_y) / tf.sqrt(var_y)
#     # data = (X, Y)

#     num_data = X.shape[0]
#     input_dim = X.shape[1]
#     output_dim = Y.shape[1]
#     num_inducing = int(np.ceil(np.log(num_data)**input_dim))

#     expert_inducing_variable = init_inducing_variables(X, num_inducing)
#     expert_inducing_variable_2 = init_inducing_variables(X, num_inducing)
#     expert_inducing_variables = [
#         expert_inducing_variable, expert_inducing_variable_2
#     ]
#     gating_inducing_variable = init_inducing_variables(X, num_inducing)

#     expert_mean_functions, expert_kernels, expert_noise_vars = init_experts(
#         num_experts=2,
#         # noise_vars=[0.005, 0.03],
#         input_dim=input_dim,
#         output_dim=output_dim)
#     # experts = ExpertsShared(expert_inducing_variable,
#     #                         output_dim,
#     #                         expert_kernels,
#     #                         expert_mean_functions,
#     #                         expert_noise_vars,
#     #                         noise_vars_trainable=[True, True])
#     experts = ExpertsSeparate(expert_inducing_variables,
#                               output_dim,
#                               expert_kernels,
#                               expert_mean_functions,
#                               expert_noise_vars,
#                               noise_vars_trainable=[True, True])
#     # noise_vars_trainable=[True, False])

#     q_mu_gating = np.zeros(
#         (num_inducing, 1)) + np.random.randn(num_inducing, 1)
#     q_sqrt_gating = np.array(
#         [10 * np.eye(num_inducing, dtype=default_float()) for _ in range(1)])
#     # q_mu_gating = init_variational_parameters(num_inducing,
#     #                                           q_mu_gating,
#     #                                           q_sqrt_gating,
#     #                                           num_latent_gps=1)
#     lengthscales_gating = tf.convert_to_tensor([1.0] * input_dim,
#                                                dtype=default_float())
#     gating_kernel = gpf.kernels.SquaredExponential(
#         lengthscales=lengthscales_gating)
#     gating_mean_function = gpf.mean_functions.Zero()

#     gating_network = GatingNetwork(gating_kernel,
#                                    gating_inducing_variable,
#                                    gating_mean_function,
#                                    num_latent_gps=1,
#                                    q_mu=q_mu_gating,
#                                    q_sqrt=q_sqrt_gating,
#                                    num_data=num_data)

#     m = SVMoGPE(input_dim,
#                 output_dim,
#                 num_inducing,
#                 experts,
#                 gating_network,
#                 minibatch_size=None,
#                 noise_var_trainable=True)

#     # TensorFlow re-traces & compiles a `tf.function`-wrapped method at *every* call if the arguments are numpy arrays instead of tf.Tensors. Hence:
#     tensor_data = tuple(map(tf.convert_to_tensor, data))
#     # elbo(tensor_data)  # run it once to trace & compile

#     minibatch_size = 100

#     train_dataset = tf.data.Dataset.from_tensor_slices(
#         (X, Y)).repeat().shuffle(num_data)

#     train_iter = iter(train_dataset.batch(minibatch_size))

#     # We turn off training for inducing point locations
#     # gpf.set_trainable(m.gating_network.inducing_variable, False)
#     # maxiter = ci_niter(20000)
#     maxiter = ci_niter(15000)
#     # maxiter = ci_niter(5000)

#     print('Model before training')
#     print_summary(m)
#     # print_summary(m.experts)
#     # print_summary(m.gating_network)

#     logf = run_adam(m, train_dataset, minibatch_size, maxiter)

#     print('Model after training')
#     print_summary(m)

#     import datetime
#     from pathlib import Path
#     date = datetime.datetime.now()
#     date_str = str(date.day) + "-" + str(date.month) + "/" + str(
#         date.hour) + str(date.minute) + "/"

#     img_save_path = "../images/model/" + date_str
#     model_save_path = "./saved_models/" + date_str
#     Path(model_save_path).mkdir(parents=True, exist_ok=True)
#     Path(img_save_path).mkdir(parents=True, exist_ok=True)

#     # print(logf)
#     # print(logf.shape)
#     # plot_and_save(m, X.numpy(), Y.numpy(), logf=logf, save_path=img_save_path)
