Mixture of Gaussian Process Experts using SVI documentation!
============================================================
This package implements a mixture of Gaussian process (GPs)
experts method where both the experts and the gating network are implemented using GPs.
The model leverages `GPflow <https://www.gpflow.org/>`_/`TensorFlow <https://www.tensorflow.org/>`_
and exploits the factorization achieved
with sparse GPs to train the model with stochastic variational inference.
More detail about the model and inference can be found in the associated paper.

Contents:

.. toctree::
   :maxdepth: 10

   getting-started
   examples
   ..
      commands

   api




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
