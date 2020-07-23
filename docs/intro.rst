############
Introduction
############
This package provides the building blocks for implementing mixture of Gaussian process experts models
using `GPflow <https://www.gpflow.org/>`_ and `TensorFlow <https://www.tensorflow.org/>`_.
It was originally created by Aidan Scannell.
It implements a method where both the experts and the gating network are formulated using
sparse variational Gaussian processes and trains the model using stochastic variational inference.
More details about the model and inference can be found at
:ref:`understanding_math:The Maths` and more details on the code base at :ref:`understanding_code:The Code`.


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
