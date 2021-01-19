=========================
The Maths
=========================
Let's recap the maths underpinning mixture of expert models as well as the extension to
Gaussian process experts (for regression).
We detail the models from a probabilistic perspective and then detail our approach to inference for
models with both the experts and gating networks based on sparse GPs (variational inference).

Model
-----

Mixture Of Experts
^^^^^^^^^^^^^^^^^^

Given a set of observations :math:`\mathcal{D} = \{ ( \mathbf{x}_n, \mathbf{y}_n ) \}_{n=1}^N`  with
inputs :math:`\mathbf{X} \in \mathbb{R}^{N\times D}` and outputs :math:`\mathbf{Y} \in \mathbb{R}^{N\times F}`
the mixture of experts (ME) marginal likelihood is given by,

.. math::


   \begin{align*} \label{eq-moe-likelihood}
   p(\mathbf{Y} | \mathbf{X}) = \prod_{n=1}^N \sum_{k=1}^K
   \underbrace{\Pr(\alpha_n=k | \mathbf{x}_n)}_{\text{Mixing Probability}}
   \underbrace{p(\mathbf{y}_n | \alpha_n=k, \mathbf{x}_n)}_{\text{Expert}}
   \end{align*}

where :math:`\alpha_n \in \{1,...K\}` is the expert indicator variable assigning the :math:`n^{\text{th}}` observation to an expert.


Mixture Of Gaussian Process Experts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Modelling the experts as GPs gives rise to a class of powerful models known as
mixture of GP experts.
Under the standard Gaussian likelihood model each expert is given by,

.. math::

   \begin{align}
   &y_n = f^{(k)}(\mathbf{x}_n) + \epsilon^{(k)}, & \epsilon^{(k)} \sim \mathcal{N}(0, (\sigma^{(k)})^2)
   \end{align}

Let's denote the :math:`k^{\text{th}}` experts latent function evaluated at :math:`\mathbf{x}_n` as
:math:`\mathbf{f}_n^{(k)} = f^{(k)}(\mathbf{x}_n)`
and at all data points as :math:`\mathbf{F}^{(k)} \in \mathbb{R}^{N \times F}`.
Placing a GP prior on each experts latent function
leads to each expert taking the form of a standard GP regression model,

.. math::

   \begin{equation*}
   p(\mathbf{y}_n | \alpha_n=k, \mathbf{x}_n)
   = \int p(\mathbf{y}_n | \mathbf{f}_n^{(k)}, \alpha_n=k) p(\mathbf{f}_{n}^{(k)} | \mathbf{F}_{\neg n}^{(k)}, \mathbf{X}) \text{d}\mathbf{f}_n^{(k)},
   \end{equation*}

where :math:`\mathbf{F}^{(k)}_{\neg n}` denotes the set :math:`\mathbf{F}^{(k)}` with the :math:`n^{\text{th}}` element removed.
Note that :math:`p(\mathbf{f}_{n}^{(k)} \mid \mathbf{F}_{\neg n}^{(k)}, \mathbf{X})` is just a GP conditional and
:math:`p(\mathbf{y}_n | \mathbf{f}_n^{(k)}, \alpha_n=k)` a Gaussian likelihood.
The mixture of GP experts marginal likelihood now takes the form,

.. math::

   \begin{align*}
   p(\mathbf{Y} | \mathbf{X})
   =\prod_{n=1}^N \sum_{k=1}^K
   \underbrace{\Pr(\alpha_n=k | \mathbf{x}_n)}_{\text{Mixing Probability}}
   \underbrace{\int p(\mathbf{y}_n | \mathbf{f}_n^{(k)}, \alpha_n=k) p(\mathbf{f}_{n}^{(k)} | \mathbf{F}_{\neg n}^{(k)}, \mathbf{X}) \text{d}\mathbf{f}_n^{(k)}}_{\text{Expert = Standard GP Regression}},
   \end{align*}

Calculating this marginal likelihood requires evaluating a GP conditional for each expert and
each data point resulting in time complexity :math:`\mathcal{O}(KN^4)`.
Many different gating networks and accompanying inference algorithms have been proposed in the literature
that attempt to address this issue.


Gating Networks
^^^^^^^^^^^^^^^
Our work is motivated by gating networks based on GPs.
Let's now detail the formulation of such a gating network.
Mixing probabilities are obtained by evaluating :math:`K` latent gating functions :math:`\{h^{(k)}\}_{k=1}^K`
and normalising their output to get
:math:`\Pr(\alpha_n=k | {h}_n^{(k)}, \mathbf{x}_n) = \frac{1}{Z}h^{(k)}(\mathbf{x}_n)`.
Each gating function :math:`h^{(k)}` describes how the expert mixing probability varies over the input space.
For each gating function :math:`h^{(k)}` let's denote :math:`h^{(k)}_n = h^{(k)}(\mathbf{x}_n)`
and :math:`\mathbf{h}^{(k)} \in R^{N \times 1}` and collect them as :math:`\mathbf{h}`.

Independent GP priors can then be placed on the latent gating functions,

.. math::

   p(\mathbf{h} | \mathbf{X}) = \prod_{k=1}^K \mathcal{N}\left(\mathbf{h}^{(k)} \mid \mu_h^{(k)}(\mathbf{X}), k_h^{(k)}(\mathbf{X}, \mathbf{X})\right),

to encode prior knowledge through the choice of mean and covariance functions.
The resulting marginal likelihood can be written with the same factorisation as the original ME
marginal likelihood,

.. math::
   \begin{align*} \label{eq-our-marginal-likelihood}
   p(\mathbf{Y} | \mathbf{X})
   =\prod_{n=1}^N \sum_{k=1}^K
   &\underbrace{\int \Pr( \alpha_n=k | {h}_{n}^{(k)}, \mathbf{x}_n) p(h_n^{(k)} | \mathbf{h}_{\neg n}, \mathbf{X}) \text{d} {h}_{n}^{(k)}}_{\text{Mixing Probability}}  \\
   &\underbrace{\int p(\mathbf{y}_n | \mathbf{f}_n^{(k)}, \alpha_n=k) p(\mathbf{f}_{n}^{(k)} | \mathbf{F}_{\neg n}^{(k)}, \mathbf{X}) \text{d}\mathbf{f}_n^{(k)}}_{\text{Expert = Standard GP Regression}},
   \end{align*}

where :math:`p(\mathbf{f}_{n}^{(k)} | \mathbf{F}_{\neg n}^{(k)}, \mathbf{X})` and :math:`p(h_n^{(k)} | \mathbf{h}_{\neg n}, \mathbf{X})`
are just GP conditionals.

Two Experts
"""""""""""
Instantiating the model with two experts is a special case for two reasons,

1. We only need one gating function as we know that :math:`\Pr(\alpha_n=2 | h_n) = 1 - \Pr(\alpha_n=1 | h_n)`,
      The output of a function :math:`h_n = h(\mathbf{x}_n)` can be mapped through a sigmoid
      function :math:`\text{sig} : \mathbb{R} \rightarrow [0, 1]` and interpreted as
      a probability :math:`\Pr(\alpha_n=1 \mid h_n)`.
      If this sigmoid function satisfies the point symmetry condition then
      we know that :math:`\Pr(\alpha_n=2 | h_n) = 1 - \Pr(\alpha_n=1 | h_n)`.
2. We can analytically marginalise :math:`\mathbf{h}`,
      We note that :math:`p({h}_n^{(k)} | \mathbf{h}_{\neg n}, \mathbf{X})`
      is a GP conditional and denote its mean :math:`\mu_h` and variance :math:`\sigma^2_h`.
      Choosing the sigmoid as the Gaussian cdf
      :math:`\Phi(h_n) = \int^{h_n}_{-\infty} \mathcal{N}(\tau | 0, 1) \text{d} \tau`
      leads to,

      .. math::

        \begin{align}
        \Pr(\alpha_n=1 | \mathbf{X}) &=
        \int \Phi({h}_n) \mathcal{N}(h_n \mid \mu_h, \sigma^2_h) \text{d} {h}_n
        = \Phi \left(\frac{\mu_{h}}{\sqrt{1 + \sigma^2_{h} }}\right).
        \end{align}



Inference
---------

Two Experts
^^^^^^^^^^^

We denote the :math:`M` inducing inputs and outputs associated with expert :math:`k` as
:math:`\mathbf{Z}_f^{(k)} \in \mathbb{R}^{M\times D}` and
:math:`\mathbf{U}_f^{(k)} \in \mathbb{R}^{M\times F}` respectively and collect them into
:math:`\mathbf{Z}_f` and :math:`\mathbf{U}_f`.
Similarly for the gating function we denote
:math:`\mathbf{Z}_h \in \mathbb{R}^{M\times D}` and
:math:`\mathbf{U}_h \in \mathbb{R}^{M\times 1}`.
Following standard sparse GP methodologies we obtain the conditionals
(:math:`p(\mathbf{f}_n^{(k)} \mid \mathbf{U}_f^{(k)}, \mathbf{x}_n)`
and :math:`p(\mathbf{h}_n \mid \mathbf{U}_h, \mathbf{x}_n)`)
which are factorized across data given the inducing points.
A central assumption is that given enough well
placed inducing points (:math:`\mathbf{U}_f` and :math:`\mathbf{U}_h`) they are a
sufficient statistic for their associated latent
function values (:math:`\mathbf{F}` and :math:`\mathbf{h}`).
Importantly, this leads to each expert being factorized across data given its inducing points,

.. math::

   \begin{align} \label{eq-experts-expectation}
   p(\mathbf{y}_n \mid \alpha_n=k, \mathbf{U}_f^{(k)}, \mathbf{x}_n) = \left\langle p(\mathbf{y}_n \mid \mathbf{f}_n^{(k)}, \alpha_n=k) \right\rangle_{p(\mathbf{f}_n^{(k)} \mid \mathbf{U}_f^{(k)}, \mathbf{x}_n)},
   \end{align}

where :math:`\left\langle \cdot \right\rangle_{p(x)}` denotes an expectation under :math:`p(x)`.
Similarly for the gating network the :math:`n^{\text{th}}` mixing probability is given by,

.. math::
   \Pr(\alpha_n=1 \mid \mathbf{U}_h, \mathbf{x}_n) = \left\langle \Phi( {h}_n) \right\rangle_{p({h}_n \mid \mathbf{U}_h, \mathbf{x}_n)}.

Denoting the inducing point distribution,

.. math::

   p(\mathbf{U}\mid\mathbf{Z}) = p(\mathbf{U}_h \mid \mathbf{Z}_h) \prod_{k=1}^K p(\mathbf{U}_f^{(k)} \mid \mathbf{Z}_f^{(k)}),

the new expanded marginal likelihood can be written as,

.. math::

   \begin{align} \label{eq-sparse-marginal-likelihood-fact}
   p(\mathbf{Y} \mid \mathbf{X})
   & = \left\langle \prod_{n=1}^N \sum_{k=1}^K \left( p(\mathbf{y}_n \mid \alpha_n=k, \mathbf{U}_f^{(k)}, \mathbf{x}_n) \Pr(\alpha_n=k \mid \mathbf{U}_h, \mathbf{x}_n) \right) \right\rangle_{p(\mathbf{U}\mid\mathbf{Z})},
   \end{align}

which has the same factorization within the expectation as in the original
ME marginal likelihood.
Let's assume that each GPs inducing points are independent and introduce
the variational distribution,

.. math::
   q(\mathbf{U}) = q(\mathbf{U}_h) \prod_{k=1}^K q(\mathbf{U}_f^{(k)}).

We know that the optimal distribution for each :math:`U` is Gaussian so we parameterise
:math:`q(\mathbf{U}_h) = \mathcal{N}(\mathbf{m}, \mathbf{S})` and
:math:`q(\mathbf{U}_f^{(k)}) = \mathcal{N}(\mathbf{m}^{(k)}, \mathbf{S}^{(k)})`.
We now use this variational distribution and Jensen's inequality to lower bound the
log marginal likelihood,

.. math::

   \begin{align*} \label{eq-lower-bound-fact}
   \text{log} p(\mathbf{Y} \mid \mathbf{X})
   & \geq \sum_{n=1}^N \left\langle \text{log} \sum_{k=1}^K p(\mathbf{y}_n \mid \alpha_n=k, \mathbf{U}_f^{(k)}, \mathbf{x}_n) \Pr(\alpha_n=k \mid \mathbf{U}_h, \mathbf{x}_n) \right\rangle_{q(\mathbf{U}_h)\prod_{k=1}^K q(\mathbf{U}_f^{(k)})} \\
   & - \sum_{k=1}^K \text{KL}\left( q(\mathbf{U}_f^{(k)}) \mid\mid p(\mathbf{U}_f^{(k)} \mid \mathbf{Z}_f^{(k)}) \right) \\
   & - \text{KL} \left( q(\mathbf{U}_h) \mid\mid p(\mathbf{U}_h \mid \mathbf{Z}_h) \right).
   \end{align*}

The key property of this lower bound is that it can be written as a sum of :math:`N` terms, each
corresponding to one observation :math:`(\mathbf{x}_n, \mathbf{y}_n)`.
We have induced the necessary factorization to perform stochastic gradient methods on the bound.

K Experts
^^^^^^^^^^^
The extension to K experts requires K gating functions in the gating network and a method for normalising their
output to get the mixing probabilities. Our inference introduces inducing points for each gating function and the
derivation of the bound is trivial given the two expert case (so we do not detail it here). 
