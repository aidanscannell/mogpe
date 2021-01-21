mogpe documentation!
============================================================

This package implements a Mixtures of Gaussian Process Experts (MoGPE) model with a GP-based gating network.
Inference exploits factorization through sparse GPs and trains a variational lower bound stochastically.
It also provides the building blocks for implementing other Mixtures of Gaussian Process Experts models.
:mod:`mogpe` uses
`GPflow 2.1 <https://github.com/GPflow/GPflow.git>`_\/`TensorFlow 2.4+ <https://github.com/tensorflow/tensorflow.git>`_
for running computations, which allows fast execution on GPUs, and uses Python ≥ 3.8.
It was originally created by `Aidan Scannell <https://www.aidanscannell.com>`_.

Getting Started
---------------
To get started please see the :ref:`getting_started:Install` instructions.
Notes on using :mod:`mogpe` can be found in :ref:`getting_started:Usage` and
the `examples directory <https://github.com/aidanscannell/mogpe/tree/master/examples>`_
and notebooks show how the model can be configured and trained.
Details on the implementation can be found in
:ref:`understanding_code:What's going on with this code?!` and the :ref:`api:mogpe API`.
The underlying mathematics of the model and inference is detailed in
:ref:`understanding_math:Hit me up with some sweet maths!`.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started

.. toctree::
   :maxdepth: 1
   :caption: Notebooks
   :hidden:

   notebooks/train_mcycle_with_2_experts

.. toctree::
   :maxdepth: 1
   :caption: Understanding
   :hidden:

   understanding_code
   understanding_math


.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   api




..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
