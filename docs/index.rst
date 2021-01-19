mogpe documentation!
============================================================

This package implements a Mixtures of Gaussian Process Experts (MoGPE) model with a GP-based gating network.
Inference exploits factorization through sparse GPs and trains a variational lower bound stochastically.
It also provides the building blocks for implementing other Mixtures of Gaussian Process Experts models.
mogpe uses GPflow 2.0/TensorFlow 2.1+ for running computations, which allows fast execution on GPUs,
and uses Python ≥ 3.6. It was originally created by Aidan Scannell.

Getting Started
^^^^^^^^^^^^^^^
Please see :ref:`understanding_code:The Code` and the :ref:`api:mogpe API` for details on the implementation and see
:ref:`understanding_math:The Maths` to get an understanding of the underlying mathematics.
For example usage see the `examples directory <https://github.com/aidanscannell/mogpe/tree/master/examples>`_
and the notebooks.

.. toctree::
   :maxdepth: 1
   :caption: Understanding:

   understanding_code
   understanding_math

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   notebooks/mixture-of-two-gp-experts
   notebooks/mixture-of-k-gp-experts

.. toctree::
   :maxdepth: 1
   :caption: API:

   api




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
