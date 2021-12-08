################################
What's going on with this code?!
################################
In this section we provide details on the Mixtures of Gaussian Process Experts code (:mod:`mogpe`).
The implementation is motivated by making it easy to implement different Mixtures of Gaussian Process 
Experts models and inference algorithms.
It exploits both inheritance and composition (building blocks of OOP)
making it easier to evolve as new features are added or requirements change.


Class Inheritance and Composition
---------------------------------------------
Let's detail the basic building blocks and how they are related.
There are three main components,

1. The mixture of experts model (:mod:`mogpe.mixture_of_experts`),
2. The set of experts (:mod:`mogpe.experts`),

   * And individual experts,
3. The gating network (:mod:`mogpe.gating_networks`),

   * And individual gating functions.


Mixture of Experts Base
^^^^^^^^^^^^^^^^^^^^^^^
At the heart of this package is the :class:`~mogpe.mixture_of_experts.MixtureOfExperts` base class
that extends GPflow's :class:`BayesianModel` class
(any instantiation requires the :func:`maximum_log_likelihood_objective` method to be implemented).
It defines the basic methods of a mixture of experts model, namely,

1. A method to predict the mixing probabilities at a set of input locations :meth:`.MixtureOfExperts.predict_mixing_probs`,
2. A method to predict the set of expert predictions at a set of input locations :meth:`.MixtureOfExperts.predict_experts_dists`,
3. A method to predict the mixture distribution at a set of input locations :meth:`.MixtureOfExperts.predict_y`.

The constructor requires an instance of a subclass of :class:`~mogpe.experts.ExpertsBase` to
represent the set of experts and an instance of a subclass of
:class:`~mogpe.gating_networks.GatingNetworkBase` to represent the gating network.

MixtureOfSVGPExperts
~~~~~~~~~~~~~~~~~~~~

The main model class in this package is :class:`~mogpe.mixture_of_experts.MixtureOfSVGPExperts`
which implements a lower bound
:func:`~mogpe.mixture_of_experts.MixtureOfSVGPExperts.maximum_log_likelihood_objective` given both
the experts and gating functions are modelled as sparse variational Gaussian processes (SVGP).
The implementation extends the :class:`~mogpe.experts.ExpertsBase` class creating
:class:`~mogpe.experts.SVGPExperts` which implements the required abstract methods as well as extra methods which are used
in the lower bound.
It also extends the :class:`~mogpe.gating_networks.GatingNetworkBase` class creating the
:class:`~mogpe.gating_networks.SVGPGatingNetwork` class.
This class implements a gating network based on SVGP's for both the special *two* expert case and
the general *k* expert case.
Let's now detail the base classes for the experts and gating network.


Expert(s) Base
^^^^^^^^^^^^^^
Before detailing the :class:`~mogpe.experts.ExpertsBase` class we need to first introduce
the base class for an individual expert.
Any class representing an individual expert must inherit the :class:`~mogpe.experts.ExpertBase`
class and implement the :func:`predict_dist` method, returning the experts prediction at Xnew.
For example, the :class:`~mogpe.experts.SVGPExpert` class inherits the
:class:`~mogpe.experts.ExpertBase` class to implement
an expert as a sparse variational Gaussian process.

Any class representing the set of all experts must inherit the
:class:`~mogpe.experts.ExpertsBase` class and should implement the :func:`predict_dists`
method, returning a batched TensorFlow Probability Distribution.
The constructor requires a list of expert instances inherited from a subclass of
:class:`~mogpe.experts.ExpertBase`.
For example, the :class:`~mogpe.experts.SVGPExperts` class represents a set of
:class:`~mogpe.experts.SVGPExpert` experts and adds a method for returning the set of
inducing point KL divergences required in the :class:`~mogpe.mixture_of_experts.MixtureOfSVGPExperts`
lower bound.

Gating Network Base
^^^^^^^^^^^^^^^^^^^
All gating networks should inherit the :class:`~.GatingNetworkBase` class and implement the
:meth:`~.GatingNetworkBase.predict_mixing_probs` and :meth:`~.GatingNetworkBase.predict_fs` methods.
This package is mainly interested in gating networks based on Gaussian processes, in particular
sparse variational Gaussian processes.
The :class:`~.SVGPGatingNetwork` class implements a gating network as a sparse variational Gaussian
process.
Similarly to GPflow's `SVGP`, its constructor requires a likelihood.
This likelihood governs the behaviour of the gating network.
If a `Bernoulli` likelihood is passed then the gating network will use a single gating function as
as we know :math:`\Pr(\alpha=2 | x) = 1 - \Pr(\alpha=1 | x)`.
As such, the kernel and inducing variables should correspond a single-output SVGP.
In the general case, i.e. with more than two experts, the gating network adopts a `Softmax` likelihood
which depends on a gating function for each expert.
In this setting, the kernel and inducing variables should be of multiple-output types, i.e.
`SeparateIndependent` and `SharedIndependentInducingVariables` respectively.
