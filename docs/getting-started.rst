Getting started
===============

This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.

Install
=======
This is a Python package that should be installed into a python virtualenv.
Create a new virtualenv and activate it, for example,

    mkvirtualenv --python=python3 mogpe-env
    workon mogpe-env

cd into the root of this package and install it and its dependencies with,

    pip install -r requirements.txt
