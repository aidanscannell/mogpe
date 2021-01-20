------------
Install
------------
This is a Python package that should be installed into a virtual environment.
Start by cloning the repo from Github::

    git clone https://github.com/aidanscannell/mogpe.git

The package can then be installed into a virutal environment by adding it as a local dependency.


Install with Poetry
^^^^^^^^^^^^^^^^^^^
:mod:`mogpe`'s dependencies and packaging are being managed with `Poetry <https://python-poetry.org/docs/>`_, instead of other tools such as Pipenv.
To install :mod:`mogpe` into an existing poetry environment add it as a dependency under
``[tool.poetry.dependencies]`` (in the :file:`pyproject.toml` configuration file) with the following line::

    mogpe = {path = "/path/to/mogpe"}

If you want to develop the :mod:`mogpe` codebase then set ``develop=true``::

    mogpe = {path = "/path/to/mogpe", develop=true}

The dependencies in a :file:`pyproject.toml` file are resolved and installed with::

    poetry install

If you do not require the development packages then you can opt to install without them::

    poetry install --no-dev

Running Python scripts inside Poetry Environments
"""""""""""""""""""""""""""""""""""""""""""""""""

There are multiple ways to run code with `Poetry <https://python-poetry.org/docs/>`_ and I advise checking out the documentation.
My favourite option is to spawn a shell within the virtual environment::

    poetry shell

and then python scripts can simply be run with::

    python codey_mc_code_face.py

Alternatively, you can run scripts without spawning an instance of the virtual environment with the
following command::

    poetry run python codey_mc_code_face.py

I am much preferring using Poetry, however, it does feel quite slow doing some things and annoyingly doesn't 
integrate that well with `Read the Docs <https://readthedocs.org/>`_.
A :file:`setup.py` file is still needed for building the docs on `Read the Docs <https://readthedocs.org/>`_, so
I use `Dephell <https://github.com/dephell/dephell>`_ to generate the :file:`requirements.txt` and :file:`setup.py` files from :file:`pyproject.toml`.


Install with Pip
^^^^^^^^^^^^^^^^
Create a new virtual environment and activate it, for example::

    mkvirtualenv --python=python3 mogpe-env
    workon mogpe-env

cd into the root of this package and install it and its dependencies with::

    pip install .

-----
Usage
-----
The model (and training with optional logging and checkpointing) can be configured using a TOML file. 
Please see the `examples <https://github.com/aidanscannell/mogpe/tree/master/examples>`_ directory showing
how to configure and train ``MixtureOfSVGPExperts`` on multiple data sets.
See the notebooks (`two experts <notebooks/train_mcycle_with_2_experts.html>`_
and `three experts <notebooks/train_mcycle_with_3_experts.html>`_)
for how to define and train an instance of ``MixtureOfSVGPExperts`` without configuration files.


Training
^^^^^^^^
The training directory contains methods for 
three different training loops, for saving and loading the model, and
for initialising the model (and training) from TOML config files.

Training Loops
""""""""""""""
``mogpe.training.training_loops`` contains three different training loops,

1. A simple TensorFlow training loop,
2. A monitoring tf training loop - a TensorFlow training loop with monitoring within tf.function().
   This method only monitors the model parameters and loss (elbo) and does not generate images.
3. A monitoring training loop - this loop generates images during training. The matplotlib functions
   cannot be inside the tf.function so this training loop should be slower but provide more insights.
   
To use Tensorboard cd to the logs directory and start Tensorboard::

    cd /path-to-log-dir
    tensorboard --logdir . --reload_multifile=true

Tensorboard can then be found by visiting `<http://localhost:6006/>`_ in your browser.

Saving/Loading
""""""""""""""
``mogpe.training.utils`` contains methods for loading and saving the model.
See the `examples <https://github.com/aidanscannell/mogpe/tree/master/examples>`_  for how to use.

TOML Config Parsers
"""""""""""""""""""
``mogpe.training.toml_config_parsers`` contains methods for 1) initialising the ``MixtureOfSVGPExperts``
class and 2) training it from a TOML config file. See the `examples <https://github.com/aidanscannell/mogpe/tree/master/examples>`_ for how to use the TOML config
parsers.

mogpe.helpers
^^^^^^^^^^^^^
The helpers directory contains classes to aid plotting models with 1D and 2D inputs.
These are exploited by the monitored training loops.
