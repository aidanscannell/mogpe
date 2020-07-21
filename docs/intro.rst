Introduction
============
This package implements a mixture of Gaussian process (GPs)
experts method where both the experts and the gating network are implemented using GPs.
It was originally created by Aidan Scannell.
The model leverages `GPflow <https://www.gpflow.org/>`_/`TensorFlow <https://www.tensorflow.org/>`_
and exploits the factorization achieved
with sparse GPs to train the model with stochastic variational inference.
More detail about the model and inference can be found in the associated paper.


Install
^^^^^^^
This is a Python package that should be installed into a python virtual environment.
Create a new virtualenv and activate it, for example,

.. code-block:: shell

    mkvirtualenv --python=python3 mogpe-env
    workon mogpe-env

cd into the root of this package and install it and its dependencies with,

.. code-block:: shell

    pip install -r requirements.txt

Getting Started
^^^^^^^^^^^^^^^

See notebooks.
