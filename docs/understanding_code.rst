########
The Code
########
In this section we provide details on the mixture of GP experts code (:mod:`mogpe.models`).
The implementation is motivated by making it easy to implement different mixture of GP
experts models and inference algorithms.
It exploits both inheritance and composition (building blocks of OOP)
making it easier to evolve as new features are added or requirements change.

Modifications to GPflow's Gaussian Processes
-------------------------------------------
This package reimplements some of the Gaussian process classes from `GPflow <https://www.gpflow.org/>`_.
This is because it is not desirable for them to inherit the :func:`maximum_log_likelihood_objective`
method from the :class:`BayesiaModel` class.
The mixture of experts lower bound also requires the :meth:`~.SVGPModel.predict_f` method to change.
It needs to predict the mean and variance of the latent function by sampling
the inducing point distribution as opposed analytically marginalising them.
The functions below have been modified/added to the :class:`.SVGPModel` class to accommodate this behaviour.

.. code-block:: python

    def predict_f(self,
                  Xnew: InputData,
                  num_inducing_samples: int = None,
                  full_cov=False,
                  full_output_cov=False) -> MeanAndVariance:
        """"Compute mean and (co)variance of latent function at Xnew.

        If num_inducing_samples is not None then sample inducing points instead
        of analytically integrating them. This is required in the mixture of
        experts lower bound."""
        with tf.name_scope('predict_f') as scope:
            if num_inducing_samples is None:
                q_mu = self.q_mu
                q_sqrt = self.q_sqrt
                mu, var = conditional(Xnew,
                                      self.inducing_variable,
                                      self.kernel,
                                      q_mu,
                                      q_sqrt=q_sqrt,
                                      full_cov=full_cov,
                                      white=self.whiten,
                                      full_output_cov=full_output_cov)
            else:
                q_mu = self.sample_inducing_points(num_inducing_samples)
                q_sqrt = None

                @tf.function
                def single_sample_conditional(q_mu):
                    # TODO requires my hack/fix to gpflow's separate_independent_conditional
                    return conditional(Xnew,
                                       self.inducing_variable,
                                       self.kernel,
                                       q_mu,
                                       q_sqrt=q_sqrt,
                                       full_cov=full_cov,
                                       white=self.whiten,
                                       full_output_cov=full_output_cov)

                mu, var = tf.map_fn(single_sample_conditional,
                                    q_mu,
                                    dtype=(default_float(), default_float()))
            return mu + self.mean_function(Xnew), var


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
            name='MultivariateNormalQ')
        inducing_samples = q_dist.sample(num_samples)
        return tf.transpose(inducing_samples, [0, 2, 1])


Class Inheritance and Composition
---------------------------------------------
In this section we detail the basic building blocks and how they are related.
There are three main components,

1. The overall mixture of experts model,
2. The set of experts,

   * And individual experts,
3. The gating network,

   * And individual gating functions.

The gating functions and individual experts are then built on top of (inherit) the modified GPflow Gaussian
process classes.


Mixture of Experts Base
^^^^^^^^^^^^^^^^^^^^^^^
At the heart of this package is the :class:`~mogpe.models.mixture_model.MixtureOfExperts` base class
that extends GPflow's :class:`BayesianModel` class
(any instantiation requires the :func:`maximum_log_likelihood_objective` method to be implemented).
It defines the basic methods of a mixture of experts models, namely,

1. A method to predict the mixing probabilities at a set of input locations :meth:`.MixtureOfExperts.predict_mixing_probs`,
2. A method to predict the set of expert predictions at a set of input locations :meth:`.MixtureOfExperts.predict_experts_dists`,
3. A method to predict the mixture distribution at a set of input locations :meth:`.MixtureOfExperts.predict_y`.

The constructor requires an instance of a subclass of :class:`~mogpe.models.experts.ExpertsBase` to
represent the set of experts and an instance of a subclass of
:class:`~mogpe.models.gating_network.GatingNetworkBase` to represent the gating network.
Let's now detail them.


Expert(s) Base
^^^^^^^^^^^^^^
Before detailing the :class:`~mogpe.models.experts.ExpertsBase` class we need to first introduce
the base class for an individual expert.
Any class representing an individual expert must inherit the :class:`~mogpe.models.experts.ExpertBase`
class and implement the :func:`predict_dist` method, returning the experts prediction at Xnew.
For example, the :class:`~mogpe.models.experts.SVGPExpert` class inherits both the
:class:`~mogpe.models.experts.ExpertBase` and :class:`~mogpe.models.gp.SVGPModel` classes to implement
an expert as a sparse variational Gaussian process.

Any class representing the set of all experts must inherit the
:class:`~mogpe.models.experts.ExpertsBase` class and should implement the :func:`predict_dists`
method, returning a batched TensorFlow Probability Distribution.
The constructor requires a list of expert instances inherited from a subclass of
:class:`~mogpe.models.experts.ExpertBase`.
For example, the :class:`~mogpe.models.experts.SVGPExperts` class represents a set of
:class:`~mogpe.models.experts.SVGPExpert` experts and adds a method for returning the set of
inducing point KL divergences required in the :class:`~mogpe.models.mixture_models.MixtureOfSVGPExperts`
lower bound.

Gating Network Base
^^^^^^^^^^^^^^^^^^^
All gating networks should inherit the :class:`~.GatingNetworkBase` class and implement the
:meth:`~.GatingNetworkBase.predict_mixing_probs` method.
This package is mainly interested in gating networks based on Gaussian processes, in particular
sparse variational Gaussian processes.
The :class:`~.SVGPGatingFunction` class implements a gating function as a sparse variational Gaussian
process.
The :class:`~.SVGPGatingNetworkBase` class provides a base for implementing gating networks
based on sparse variational Gaussian processes.
Its constructor requires a list of :class:`~.SVGPGatingFunction` instances, the gating functions!
It inherits :class:`~.GatingNetworkBase` and also implements
:meth:`~.SVGPGatingNetworkBase.prior_kls`, returning the set of inducing point KL divergences for
all gating functions, required in the :class:`~mogpe.models.mixture_models.MixtureOfSVGPExperts`
lower bound.

The package implements two variants of :class:`~.SVGPGatingNetworkBase`,

1. :class:`~.SVGPGatingNetworkBinary` - This class represents the special case of two experts. In this scenario only a single gating function is required as we know :math:`\Pr(\alpha=0 | x) = 1 - \Pr(\alpha=1 | x)`.
2. :class:`~.SVGPGatingNetworkMulti` - This is the general mixture of GP experts that can use K experts and K gating functions. Its constructor requires a :class:`Likelihood` whose role is to map the gating functions outputs to expert mixing probabilities e.g. softmax.


MixtureOfSVGPExperts
^^^^^^^^^^^^^^^^^^^^

The main model class in this package is :class:`~mogpe.models.mixture_model.MixtureOfSVGPExperts`
which implements a lower bound
:func:`~mogpe.models.mixture_model.MixtureOfSVGPExperts.maximum_log_likelihood_objective` given both
the experts and gating functions are modelled as sparse Gaussian processes.
Please see the :ref:`understanding_math:Derivations (Lower Bound)` section for more details
on the lower bound.
