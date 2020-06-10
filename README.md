
# Table of Contents

1.  [mixture-of-two-gp-experts](#org4818d5f)
    1.  [Install](#orgf538eff)
    2.  [Reproducing Figures](#org8e83194)
    3.  [Data Sets](#org9cc58bd)


<a id="org4818d5f"></a>

# mixture-of-two-gp-experts

This repository contains the code accompanying NeurIPS 2020 submission
*Mixture of Two Gaussian Process Experts using Stochastic Variational Inference*.

This repository contains,

1.  saved models that were used to generate the figures in the paper,
2.  the data sets,
3.  all of the code (including training scripts).

We detail how to,

1.  generate the figures in the paper from the saved models,
2.  how to train our model on both data sets (and plot).


<a id="orgf538eff"></a>

## Install

This is a Python package that should be installed into a python virtualenv.
Create a new virtualenv and activate it, for example,

    mkvirtualenv --python=python3 mogpe-env
    workon mogpe-env

cd into the root of this package and install it and its dependencies with,

    pip install -r requirements.txt


<a id="org8e83194"></a>

## Reproducing Figures

To reproduce the figures in the paper (using the models saved in [models/saved<sub>model</sub>](./models/saved_model)),

-   Figure 3a - Run [src/visualization/plot<sub>artificial</sub><sub>rbf</sub><sub>gating.py</sub>](./src/visualization/plot_artificial_rbf_gating.py) with,

        python src/visualization/plot_artificial_rbf_gating.py
-   Figure 3b - Run [src/visualization/plot<sub>artificial</sub><sub>prod</sub><sub>cosine</sub><sub>rbf</sub><sub>gating.py</sub>](./src/visualization/plot_artificial_prod_cosine_rbf_gating.py) with,

        python src/visualization/plot_artificial_prod_cosine_rbf_gating.py
-   Figure 4 - Run [src/visualization/plot<sub>mcycle.py</sub>](./src/visualization/plot_mcycle.py) with,

        python src/visualization/plot_mcycle.py

The json [config files](./configs) contain the configurations that were used to train the model for each figure.
Our model can be trained from scratch using these config files,

-   Artificial data (Figure 3)
    -   Figure 3a - Set the `json_file` variable in
        [src/models/train<sub>and</sub><sub>plot</sub><sub>artificial.py</sub>](./src/models/train_and_plot_artificial.py) to [configs/figure-3a.json](./configs/figure-3a.json)
        and then run it.
    -   Figure 3b - Set the `json_file` variable in
        [src/models/train<sub>and</sub><sub>plot</sub><sub>artificial.py</sub>](./src/models/train_and_plot_artificial.py) to [configs/figure-3b.json](./configs/figure-3b.json)
         and then run it.
-   Motorcycle data (Figure 4) - Run [src/models/train<sub>and</sub><sub>plot</sub><sub>mcycle.py</sub>](./src/models/train_and_plot_mcycle.py)

Figure 1 is not based on our model but instead provides intuition about different
gating network formulations. It was generated with
[src/visualization/figure<sub>1</sub><sub>comparing</sub><sub>gating</sub><sub>networks.py</sub>](./src/visualization/figure_1_comparing_gating_networks.py).


<a id="org9cc58bd"></a>

## Data Sets

The model is tested on two data sets,

1.  an artificial data set,
2.  the motorcycle data set.

The data sets can be found in the [data](./data) directory.
The motorcycle data set was obtained from [here](https://vincentarelbundock.github.io/Rdatasets/datasets.html) and is saved at [data/external/mcycle.csv](./data/external/mcycle.csv).
