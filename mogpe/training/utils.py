import gpflow as gpf
import numpy as np
import pathlib
import time
import tensorflow as tf

from gpflow.monitor import ImageToTensorBoard, MonitorTaskGroup, Monitor


def training_tf_loop(model,
                     training_loss,
                     epochs: int = 1,
                     num_batches_per_epoch: int = 1,
                     logging_epoch_freq: int = 100):
    """Runs Adam optimizer on model with training_loss (no monitoring).

    :param model: The model to be trained.
    :param training_loss: A function that returns the training objective.
    :param epochs: The number of full data passes (epochs).
    :param num_batches_per_epoch: The number of batches per epoch
    :param logging_epoch_freq: The epoch frequency that the training loss is printed.
    """
    optimizer = tf.optimizers.Adam()

    @tf.function
    def tf_optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    # t = time.time()
    for epoch in range(epochs):
        for _ in range(num_batches_per_epoch):
            tf_optimization_step()
            # tf_optimization_step(model, training_loss, optimizer)
        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
        # duration = t - time.time()
        # print("Iteration duration: ", duration)
        # t = time.time()


def monitored_training_tf_loop(model,
                               training_loss,
                               epochs: int = 1,
                               num_batches_per_epoch: int = 1,
                               fast_tasks: gpf.monitor.MonitorTaskGroup = None,
                               logging_epoch_freq: int = 100):
    """Monitors Adam optimizer on model with training_loss.

    Both training and monitoring are inside tf.function (no image monitoring).
    This method only monitors the fast tasks as matplotlib code cannot be built
    in a TF graph.

    :param model: The model to be trained.
    :param training_loss: A function that returns the training objective.
    :param epochs: The number of full data passes (epochs).
    :param num_batches_per_epoch: The number of batches per epoch
    :param fast_tasks: gpflow monitor fast tasks e.g.
        MonitorTaskGroup([ScalarToTensorBoard(log_dir, training_loss, "elbo")])
    :param logging_epoch_freq: The epoch frequency that the training loss is printed.
    """
    optimizer = tf.optimizers.Adam()
    monitor = Monitor(fast_tasks)

    @tf.function
    def monitored_tf_opt_step(epoch):
        optimizer.minimize(training_loss, model.trainable_variables)
        monitor(epoch)

    # t = time.time()
    epochs = tf.constant(epochs)  # needs to be tf.const
    for epoch in tf.range(epochs):
        for _ in range(num_batches_per_epoch):
            monitored_tf_opt_step(epoch)
        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
            # duration = t - time.time()
            # print("Iteration duration: ", duration)
            # t = time.time()


def monitored_training_loop(model,
                            training_loss,
                            epochs: int = 1,
                            num_batches_per_epoch: int = 1,
                            fast_tasks: gpf.monitor.MonitorTaskGroup = None,
                            slow_tasks: gpf.monitor.MonitorTaskGroup = None,
                            logging_epoch_freq: int = 100,
                            save_dir: str=""):
    """Monitors (with images) Adam optimizer on model with training_loss.

    Monitoring is not inside tf.function so this method will be slower than
    monitored_training_tf_loop.

    :param model: The model to be trained.
    :param training_loss: A function that returns the training objective.
    :param epochs: The number of full data passes (epochs).
    :param num_batches_per_epoch: The number of batches per epoch
    :param fast_tasks: gpflow monitor fast tasks e.g.
        MonitorTaskGroup([ScalarToTensorBoard(log_dir, training_loss, "elbo")])
    :param slow_tasks: gpflow monitor slow tasks e.g. plotting images
    :param logging_epoch_freq: The epoch frequency that the training loss is printed.
    """
    optimizer = tf.optimizers.Adam()
    # checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    @tf.function
    def tf_optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    monitor = Monitor(fast_tasks, slow_tasks)

    t = time.time()
    for epoch in range(epochs):
        for _ in range(num_batches_per_epoch):
            tf_optimization_step()
            # duration = t - time.time()
            # print("Iteration duration: ", duration)
            # t = time.time()
        monitor(epoch)
        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
            # save_model(model, save_model_dir)
        # duration = t - time.time()
        # print("Iteration duration: ", duration)
        # t = time.time()


def save_model(model, save_dir=None):
    if save_dir is None:
        save_dir = str(pathlib.Path(tempfile.gettempdir()))
    params = gpf.utilities.parameter_dict(model)
    gpf.utilities.multiple_assign(model, params)

    frozen_model = gpf.utilities.freeze(model)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)])
    def predict_experts_ys(Xnew):
        dists = frozen_model.predict_experts_dists(Xnew)
        return dists.mean(), dists.variance()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)])
    def predict_experts_fs(Xnew):
        return frozen_model.predict_experts_fs(Xnew)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)])
    def predict_gating_fs(Xnew):
        return frozen_model.gating_network.predict_fs(Xnew)

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)])
    # def predict_y(Xnew):
    #     dist = frozen_model.predict_y(Xnew)
    #     return dist

    # @tf.function(input_signature=[tf.TensorSpec(shape=[3, 2], dtype=tf.float64), tf.TensorSpec([], dtype=tf.int64)])
    # def predict_y_samples(Xnew, num_samples):
    #     # return frozen_model.predict_y_samples(Xnew, num_samples)
    #     return frozen_model.predict_y(Xnew).sample(num_samples)

    # TODO Needs to be fixed number of data points or ndiag_mc fails
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float64)])
    def predict_mixing_probs(Xnew):
        probs = frozen_model.predict_mixing_probs(Xnew, num_inducing_samples=None)
        return probs

    module_to_save = tf.Module()
    # predict_y_fn = tf.function(
    #     frozen_model.predict_y,
    #     input_signature=[tf.TensorSpec(shape=[1,2], dtype=tf.float64)],
    #     # input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    #     autograph=False,
    # )
    # sample_y_fn = tf.function(
    #     frozen_model.predict_y_samples,
    #     input_signature=[tf.TensorSpec(shape=[1,2], dtype=tf.float64)],
    #     # input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    #     autograph=False,
    # )
    # predict_experts_fs_fn = tf.function(
    #     frozen_model.predict_experts_fs,
    #     input_signature=[tf.TensorSpec(shape=[1,2], dtype=tf.float64)],
    #     # input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    #     autograph=False,
    # )
    # predict_experts_ys_fn = tf.function(
    #     predict_experts_ys,
    #     input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)],
    #     # input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    #     autograph=False,
    # )
    # predict_gating_fs_fn = tf.function(
    #     frozen_model.gating_network.predict_fs,
    #     input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float64)],
    #     # input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    #     autograph=False,
    # )
    # predict_mixing_probs_fn = tf.function(
    #     frozen_model.predict_mixing_probs,
    #     input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float64)],
    #     # input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    #     autograph=False,
    # )
    # module_to_save.predict_y= predict_y_fn
    # module_to_save.predict_y_samles = sample_y_fn
    # module_to_save.predict_experts_fs = predict_experts_fs_fn
    # module_to_save.predict_experts_ys = predict_experts_ys_fn
    # module_to_save.predict_gating_fs = predict_gating_fs_fn
    # module_to_save.predict_mixing_probs = predict_mixing_probs_fn

    module_to_save.predict_experts_ys = predict_experts_ys
    module_to_save.predict_experts_fs = predict_experts_fs
    module_to_save.predict_gating_fs = predict_gating_fs
    # module_to_save.predict_y= predict_y
    # module_to_save.predict_y_samples = predict_y_samples
    module_to_save.predict_mixing_probs = predict_mixing_probs

    tf.saved_model.save(module_to_save, save_dir)


def save_model_params(model, save_dir):
    print('Saving param dict to: ', save_dir)
    param_dict = gpf.utilities.parameter_dict(model)
    print(param_dict)
    np.savez(save_dir, **param_dict)


def load_model_params(load_dir):
    params = np.load(load_dir, allow_pickle=True)
    print('loading params you')
    print(params)
    for key in params.keys():
        print(key)
    # print(params['.experts.experts_list[0].mean_function.c'])
    q_mu = params['.experts.experts_list[0].q_mu']
    print(params['.gating_network.gating_function_list[0].kernel.kernels[0].variance'])
    # print(params['q_mu'])


if __name__ == "__main__":
    load_dir = '/Users/aidanscannell/Developer/python-projects/mixture-of-k-gp-experts/models/logs/quadcopter/09-02-151153-param_dict.npz'
    load_model_params(load_dir)
