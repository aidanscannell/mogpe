
# Table of Contents

1.  [Saved Models](#org4c6a275)
    1.  [Artificial Data](#org23f495c)
    2.  [Motorcycle Data](#orge90717e)


<a id="org4c6a275"></a>

# Saved Models

This directory contains the saved models that were used to generate the figures in the paper.


<a id="org23f495c"></a>

## Artificial Data

Expert one has a cosine kernel and expert two has a squared exponential kernel.

-   [artificial/rbf<sub>gating</sub><sub>kernel</sub>](./artificial/rbf_gating_kernel) is our model trained on the artificial data set using a squared exponential
    covariance function for the gating network GP (Figure 3a). This is trained with the config in
    [../../configs/figure-3a.json](../../configs/figure-3a.json).
-   [./artificial/prod<sub>rbf</sub><sub>cosine</sub><sub>kernel</sub>](./artificial/prod_rbf_cosine_kernel) is our model trained on the artificial data set using the product
    of a squared exponential kernel and a cosine kernel as the covariance function
    for the gating network GP (Figure 3b). This is trained with the config in ([../../configs/figure-3b.json](../../configs/figure-3b.json)).


<a id="orge90717e"></a>

## Motorcycle Data

-   [./mcycle](./mcycle) is our model trained on the motorcycle data set using
    squared exponential kernels for all GPs (Figure 4).
    This is trained with the config in ([../../configs/figure-4-mcycle.json](../../configs/figure-4-mcycle.json)).
-   `svgp_cycle.npz` is the result of training a sparse variational GP (SVGP) on the motorcycle data set.
    It is used in [../../src/visualization](../../src/visualization) for generating plots comparing our method to a SVGP.
