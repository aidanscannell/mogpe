import gpflow as gpf
import numpy as np
import pandas as pd
import pathlib
import tempfile
import tensorflow as tf
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive, triangular


def run_adam(model, train_dataset, minibatch_size, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf


def init_inducing_variables(X, num_inducing):
    num_data = X.shape[0]
    input_dim = X.shape[1]
    idx = np.random.choice(range(num_data), size=num_inducing, replace=False)
    if type(X) is np.ndarray:
        inducing_inputs = X[idx, ...].reshape(-1, input_dim)
    else:
        inducing_inputs = X.numpy()[idx, ...].reshape(-1, input_dim)
    return inducing_inputs


def init_variational_parameters(num_inducing,
                                q_mu=None,
                                q_sqrt=None,
                                q_diag=False,
                                num_latent_gps=1):
    q_mu = np.zeros((num_inducing, num_latent_gps)) if q_mu is None else q_mu
    q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

    if q_sqrt is None:
        if q_diag:
            ones = np.ones((num_inducing, num_latent_gps),
                           dtype=default_float())
            q_sqrt = Parameter(ones, transform=positive())  # [M, P]
        else:
            q_sqrt = [
                np.eye(num_inducing, dtype=default_float())
                for _ in range(num_latent_gps)
            ]
            q_sqrt = np.array(q_sqrt)
            q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
    else:
        if q_diag:
            assert q_sqrt.ndim == 2
            num_latent_gps = q_sqrt.shape[1]
            q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
        else:
            assert q_sqrt.ndim == 3
            num_latent_gps = q_sqrt.shape[0]
            num_inducing = q_sqrt.shape[1]
            q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]
    return q_mu, q_sqrt


# def init_experts(num_experts=2,
#                  noise_vars=[0.005, 0.3],
#                  input_dim=1,
#                  output_dim=1):
#     # init expert kernels and mean functions
#     expert_mean_functions = []
#     expert_kernels = []
#     expert_noise_vars = []

#     for k in range(num_experts):
#         expert_mean_functions.append(gpf.mean_functions.Constant())
#         # Create list of kernels for each output
#         lengthscales_gating = tf.convert_to_tensor([1.0] * input_dim,
#                                                    dtype=default_float())

#         expert_noise_vars.append(
#             tf.convert_to_tensor([noise_vars[k]] * output_dim,
#                                  dtype=default_float()))
#         kern_list = [
#             gpf.kernels.SquaredExponential(lengthscales=lengthscales_gating)
#             for _ in range(output_dim)
#         ]
#         # Create multioutput kernel from kernel list
#         kern = gpf.kernels.SeparateIndependent(kern_list)
#         expert_kernels.append(kern)
#     return expert_mean_functions, expert_kernels, expert_noise_vars

# def init_svmogpe(X,
#                  Y,
#                  num_inducing=None,
#                  bound='tight',
#                  num_samples_expert_expectation=None):
#     from experts import ExpertsSeparate
#     from gating_network import GatingNetwork
#     from svmogpe import SVMoGPE
#     num_data = X.shape[0]
#     input_dim = X.shape[1]
#     output_dim = Y.shape[1]
#     if num_inducing is None:
#         num_inducing = int(np.ceil(np.log(num_data)**input_dim))

#     expert_inducing_variable = init_inducing_variables(X, num_inducing)
#     expert_inducing_variable_2 = init_inducing_variables(X, num_inducing)
#     expert_inducing_variables = [
#         expert_inducing_variable, expert_inducing_variable_2
#     ]
#     gating_inducing_variable = init_inducing_variables(X, num_inducing)

#     expert_mean_functions, expert_kernels, expert_noise_vars = init_experts(
#         num_experts=2,
#         noise_vars=[1e-1, 1e-1],
#         # noise_vars=[1e-3, 1e-3],
#         input_dim=input_dim,
#         output_dim=output_dim)
#     experts = ExpertsSeparate(expert_inducing_variables,
#                               output_dim,
#                               expert_kernels,
#                               expert_mean_functions,
#                               num_samples=num_samples_expert_expectation,
#                               noise_vars=expert_noise_vars)

#     q_mu_gating = np.zeros(
#         (num_inducing, 1)) + np.random.randn(num_inducing, 1) * 2
#     q_sqrt_gating = np.array(
#         [10 * np.eye(num_inducing, dtype=default_float()) for _ in range(1)])
#     lengthscales_gating = tf.convert_to_tensor([0.5] * input_dim,
#                                                dtype=default_float())
#     gating_kernel = gpf.kernels.SquaredExponential(
#         lengthscales=lengthscales_gating)
#     gating_mean_function = gpf.mean_functions.Zero()

#     gating_network = GatingNetwork(gating_kernel,
#                                    gating_inducing_variable,
#                                    gating_mean_function,
#                                    num_latent_gps=1,
#                                    q_mu=q_mu_gating,
#                                    q_sqrt=q_sqrt_gating,
#                                    num_data=num_data)

#     m = SVMoGPE(input_dim,
#                 output_dim,
#                 experts=experts,
#                 gating_network=gating_network,
#                 bound=bound)
#     return m


def standardise_data(X, Y):
    mean_x, var_x = tf.nn.moments(X, axes=[0])
    mean_y, var_y = tf.nn.moments(Y, axes=[0])
    X = (X - mean_x) / tf.sqrt(var_x)
    Y = (Y - mean_y) / tf.sqrt(var_y)
    return X, Y


def load_mixture_dataset(
        filename='../data/artificial/artificial-1d-mixture-2.npz',
        standardise=True):
    data = np.load(filename)
    X = data['x']
    Y = data['y']
    F = data['f']
    prob_a_0 = data['prob_a_0']
    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)
    if standardise:
        X, Y = standardise_data(X, Y)
    data = (X, Y)
    return data, F, prob_a_0


def load_mcycle_dataset(filename='~/Developer/datasets/mcycle.csv'):
    df = pd.read_csv(filename, sep=',')
    X = pd.to_numeric(df['times'])
    Y = pd.to_numeric(df['accel'])
    # Y = Y + 30 * np.sin(Y)
    X = X.to_numpy().reshape(-1, 1)
    Y = Y.to_numpy().reshape(-1, 1)

    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)
    # standardise input
    X, Y = standardise_data(X, Y)
    data = (X, Y)
    return data


# import matplotlib.pyplot as plt
# X, Y = load_mcycle_dataset()
# plt.scatter(X, Y)
# plt.show()


def trim_dataset(X, Y, x1_low=-3., x2_low=-3., x1_high=0., x2_high=-1.):
    mask_0 = X[:, 0] < x1_low
    mask_1 = X[:, 1] < x2_low
    mask_2 = X[:, 0] > x1_high
    mask_3 = X[:, 1] > x2_high
    mask = mask_0 | mask_1 | mask_2 | mask_3
    X_partial = X[mask, :]
    Y_partial = Y[mask, :]
    x1 = [x1_low, x1_low, x1_high, x1_high, x1_low]
    x2 = [x2_low, x2_high, x2_high, x2_low, x2_low]
    X_missing = [x1, x2]

    print("New data shape:", Y_partial.shape)
    return X_partial, Y_partial


def load_quadcopter_dataset(
        filename='../data/npz/turbulence/model_data_fan_fixed_subset.npz'):
    data = np.load(filename)
    X = data['x']
    Y = data['y']
    # Y = Y[:, 0:2]
    Y = Y[:, 0:1]
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)

    # remove some data points
    X_subset, Y_subset = trim_dataset(X,
                                      Y,
                                      x1_low=-3.,
                                      x2_low=-3.,
                                      x1_high=0.,
                                      x2_high=-1.)

    data = (X_subset, Y_subset)
    return data


def save_model(model, save_dir=None):
    if save_dir is None:
        save_dir = str(pathlib.Path(tempfile.gettempdir()))
    params = gpf.utilities.parameter_dict(model)
    gpf.utilities.multiple_assign(model, params)

    frozen_model = gpf.utilities.freeze(model)

    module_to_save = tf.Module()
    predict_y_mm_fn = tf.function(
        frozen_model.predict_y_moment_matched,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    sample_y_fn = tf.function(
        frozen_model.sample_y,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    predict_experts_fs_fn = tf.function(
        frozen_model.predict_experts_fs,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    predict_experts_ys_fn = tf.function(
        frozen_model.predict_experts_ys,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    predict_gating_h_fn = tf.function(
        frozen_model.predict_gating_h,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    predict_mixing_probs_fn = tf.function(
        frozen_model.predict_mixing_probs,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        autograph=False,
    )
    module_to_save.predict_y_moment_matched = predict_y_mm_fn
    module_to_save.sample_y = sample_y_fn
    module_to_save.predict_experts_fs = predict_experts_fs_fn
    module_to_save.predict_experts_ys = predict_experts_ys_fn
    module_to_save.predict_gating_h = predict_gating_h_fn
    module_to_save.predict_mixing_probs = predict_mixing_probs_fn

    tf.saved_model.save(module_to_save, save_dir)
