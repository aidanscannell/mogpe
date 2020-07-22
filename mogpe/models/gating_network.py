import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABC, abstractmethod
from gpflow import Module
from gpflow.conditionals import conditional, sample_conditional
from gpflow.conditionals.util import sample_mvn
from gpflow.config import default_float
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.models.util import inducingpoint_wrapper
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.likelihoods import Likelihood, Softmax, Bernoulli
from typing import List

from mogpe.models.gp import SVGPModel


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 -
                                                          2 * jitter) + jitter


class SVGPGatingFunction(SVGPModel):
    # TODO either remove likelihood or use Bernoulli/Softmax
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
        super().__init__(kernel,
                         likelihood=None,
                         inducing_variable=inducing_variable,
                         mean_function=mean_function,
                         num_latent_gps=num_latent_gps,
                         q_diag=q_diag,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt,
                         whiten=whiten,
                         num_data=num_data)


class GatingNetworkBase(Module, ABC):
    """Abstract base class for the gating network."""
    @abstractmethod
    def predict_mixing_probs(self, Xnew: InputData, **kwargs):
        """Calculates the set of experts mixing probabilities at Xnew :math:`\{\Pr(\\alpha=k | x)\}^K_{k=1}`

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: a batched Tensor with shape [..., num_test, output_dim, num_experts]
        """
        raise NotImplementedError


class SVGPGatingNetworkBase(GatingNetworkBase):
    """Abstract base class for gating networks based on SVGPs."""
    def __init__(self,
                 gating_function_list: List[SVGPGatingFunction] = None,
                 name='GatingNetwork'):
        super().__init__(name=name)
        assert isinstance(gating_function_list, List)
        for gating_function in gating_function_list:
            assert isinstance(gating_function, SVGPGatingFunction)
        self.gating_function_list = gating_function_list
        self.num_experts = len(gating_function_list)

    def prior_kls(self) -> tf.Tensor:
        """Returns the set of experts KL divergences as a batched tensor.

        :returns: a Tensor with shape [num_experts,]
        """
        kls = []
        for gating_function in self.gating_function_list:
            kls.append(gating_function.prior_kl())
        return tf.convert_to_tensor(kls)

    # @abstractmethod
    # def prior_kls(self) -> tf.Tensor:
    #     """Returns the set of experts KL divergences as a batched tensor.

    #     :returns: a Tensor with shape [num_experts,]
    #     """
    #     raise NotImplementedError


class SVGPGatingNetworkMulti(SVGPGatingNetworkBase):
    # TODO either remove likelihood or use Bernoulli/Softmax
    def __init__(self,
                 gating_function_list: List[SVGPGatingFunction] = None,
                 likelihood: Likelihood = None,
                 name='GatingNetwork'):
        super().__init__(gating_function_list, name=name)
        # assert isinstance(gating_function_list, List)
        # for gating_function in gating_function_list:
        #     assert isinstance(gating_function, SVGPGatingFunction)
        # self.gating_function_list = gating_function_list
        # self.num_experts = len(gating_function_list)

        if likelihood is None:
            self.likelihood = Softmax(num_classes=self.num_experts)
        else:
            self.likelihood = likelihood

    def predict_mixing_probs(self,
                             Xnew: InputData,
                             num_inducing_samples: int = None):
        from mogpe.models.quadrature import ndiag_mc

        mixing_probs = []
        Fmu, Fvar = [], []
        for gating_function in self.gating_function_list:
            # num_inducing_samples = None
            f_mu, f_var = gating_function.predict_f(Xnew, num_inducing_samples)
            Fmu.append(f_mu)
            Fvar.append(f_var)
        # Fmu = tf.stack(Fmu)
        # Fvar = tf.stack(Fvar)
        Fmu = tf.concat(Fmu, -1)
        Fvar = tf.concat(Fvar, -1)

        def single_predict_mean(args):
            Fmu, Fvar = args
            integrand2 = lambda *X: self.likelihood.conditional_variance(
                *X) + tf.square(self.likelihood.conditional_mean(*X))
            epsilon = None
            E_y, E_y2 = ndiag_mc(
                [self.likelihood.conditional_mean, integrand2],
                S=self.likelihood.num_monte_carlo_points,
                Fmu=Fmu,
                Fvar=Fvar,
                epsilon=epsilon)
            return E_y

        if num_inducing_samples is None:
            mixing_probs = self.likelihood.predict_mean_and_var(Fmu, Fvar)[0]
        else:
            mixing_probs = tf.map_fn(single_predict_mean, (Fmu, Fvar),
                                     dtype=tf.float64)
        return mixing_probs


class SVGPGatingNetworkBinary(SVGPGatingNetworkBase):
    def __init__(self,
                 gating_function: SVGPGatingFunction = None,
                 name='GatingNetwork'):
        assert isinstance(gating_function, SVGPGatingFunction)
        gating_function_list = [gating_function]
        super().__init__(gating_function_list, name=name)
        # self.gating_function = gating_function
        self.likelihood = Bernoulli()
        self.num_experts = 2

    # def prior_kls(self) -> tf.Tensor:
    #     """Returns the set of experts KL divergences as a batched tensor.

    #     :returns: a Tensor with shape [num_experts,]
    #     """
    #     return tf.convert_to_tensor(self.gating_function.prior_kl())

    def predict_mixing_probs(self,
                             Xnew: InputData,
                             num_inducing_samples: int = None):
        """Compute mixing probabilities.

        Returns a tensor with dimensions,
            [num_inducing_samples,num_data, output_dim, num_experts]
        if num_inducing_samples=None otherwise a tensor with dimensions,
            [num_data, output_dim, num_experts]

        .. math::
            \\mathbf{u}_h \sim \mathcal{N}(q\_mu, q\_sqrt \cdot q\_sqrt^T) \\\\
            \\Pr(\\alpha=k | \\mathbf{Xnew}, \\mathbf{u}_h)

        :param Xnew: test input(s) [num_data, input_dim]
        :param num_inducing_samples: how many samples to draw from inducing points
        """
        h_mu, h_var = self.gating_function_list[0].predict_f(
            Xnew, num_inducing_samples, full_cov=False)

        def single_predict_mean(args):
            h_mu, h_var = args
            return self.likelihood.predict_mean_and_var(h_mu, h_var)[0]

        if num_inducing_samples is None:
            prob_a_0 = self.likelihood.predict_mean_and_var(h_mu, h_var)[0]
        else:
            prob_a_0 = tf.map_fn(single_predict_mean, (h_mu, h_var),
                                 dtype=tf.float64)

        print('prob_a_0 in gating network mixing probs')
        print(prob_a_0.shape)
        prob_a_1 = 1 - prob_a_0
        mixing_probs = tf.stack([prob_a_0, prob_a_1], -1)
        return mixing_probs


def init_fake_gating_network_binary(X, Y):
    from mogpe.models.utils.model import init_inducing_variables
    output_dim = Y.shape[1]
    input_dim = X.shape[1]

    num_inducing = 7
    inducing_variable = init_inducing_variables(X, num_inducing)

    inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(inducing_variable))

    noise_var = 0.1
    lengthscale = 1.
    mean_function = gpf.mean_functions.Zero()
    likelihood = None

    q_mu = np.zeros(
        (num_inducing,
         output_dim)) + np.random.randn(num_inducing, output_dim) * 2
    q_sqrt = np.array([
        10 * np.eye(num_inducing, dtype=default_float())
        for _ in range(output_dim)
    ])

    kern_list = []
    for _ in range(output_dim):
        # Create multioutput kernel from kernel list
        lengthscale = tf.convert_to_tensor([lengthscale] * input_dim,
                                           dtype=default_float())
        kern_list.append(gpf.kernels.RBF(lengthscales=lengthscale))
    kernel = gpf.kernels.SeparateIndependent(kern_list)

    gating_function = SVGPGatingFunction(kernel,
                                         likelihood,
                                         inducing_variable,
                                         mean_function,
                                         q_mu=q_mu,
                                         q_sqrt=q_sqrt)
    return SVGPGatingNetworkBinary(gating_function)


if __name__ == "__main__":
    # Load data set
    from mogpe.data.utils import load_mixture_dataset
    data_file = '../../data/processed/artificial-data-used-in-paper.npz'
    data, F, prob_a_0 = load_mixture_dataset(filename=data_file,
                                             standardise=False)
    X, Y = data

    gating_network = init_fake_gating_network_binary(X, Y)
    # mixing_probs = gating_network.predict_mixing_probs(X, 10)
    mixing_probs = gating_network.predict_mixing_probs(X)
    print(mixing_probs.shape)
    # print(mixing_probs[0].shape)
