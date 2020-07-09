import numpy as np
from gpflow import default_float, Parameter
from gpflow.utilities import positive, triangular


def init_inducing_variables(X, num_inducing):
    num_data = X.shape[0]
    input_dim = X.shape[1]
    idx = np.random.choice(range(num_data), size=num_inducing, replace=False)
    if type(X) is np.ndarray:
        inducing_inputs = X[idx, ...].reshape(-1, input_dim)
    else:
        inducing_inputs = X.numpy()[idx, ...].reshape(-1, input_dim)
    return inducing_inputs


# def init_variational_parameters(num_inducing,
#                                 q_mu=None,
#                                 q_sqrt=None,
#                                 q_diag=False,
#                                 num_latent_gps=1):
#     q_mu = np.zeros((num_inducing, num_latent_gps)) if q_mu is None else q_mu
#     q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

#     if q_sqrt is None:
#         if q_diag:
#             ones = np.ones((num_inducing, num_latent_gps),
#                            dtype=default_float())
#             q_sqrt = Parameter(ones, transform=positive())  # [M, P]
#         else:
#             q_sqrt = [
#                 np.eye(num_inducing, dtype=default_float())
#                 for _ in range(num_latent_gps)
#             ]
#             q_sqrt = np.array(q_sqrt)
#             q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
#     else:
#         if q_diag:
#             assert q_sqrt.ndim == 2
#             num_latent_gps = q_sqrt.shape[1]
#             q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
#         else:
#             assert q_sqrt.ndim == 3
#             num_latent_gps = q_sqrt.shape[0]
#             num_inducing = q_sqrt.shape[1]
#             q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]
#     return q_mu, q_sqrt
