#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import pathlib
import pickle
import tensorflow as tf


def update_model_from_checkpoint(model, ckpt_dir):
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    final_checkpoint = manager.checkpoints[-1]
    ckpt.restore(final_checkpoint)
    print('Restored Model')
    gpf.utilities.print_summary(model)
    return model


def load_model_from_config_and_checkpoint(config_file, ckpt_dir, X=None):
    from mogpe.training import create_mosvgpe_model_from_config
    model = create_mosvgpe_model_from_config(config_file, X)
    return update_model_from_checkpoint(model, ckpt_dir)


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

    module_to_save.predict_experts_ys = predict_experts_ys
    module_to_save.predict_experts_fs = predict_experts_fs
    module_to_save.predict_gating_fs = predict_gating_fs
    # module_to_save.predict_y= predict_y
    # module_to_save.predict_y_samples = predict_y_samples
    module_to_save.predict_mixing_probs = predict_mixing_probs

    tf.saved_model.save(module_to_save, save_dir)


def save_models_param_dict(model, save_dir):
    save_model_dir = save_dir + "/param_dict.pickle"
    param_dict = gpf.utilities.parameter_dict(model)
    print('param dict')
    print(param_dict)
    f = open(save_model_dir, "wb")
    pickle.dump(param_dict, f)
    f.close()


if __name__ == "__main__":

    config_file = '../../examples/mcycle/configs/config_2_experts.toml'
    ckpt_dir = "../../examples/logs/mcycle/two_experts/11-14-164351"

    model = load_model_from_config_and_checkpoint(config_file, ckpt_dir)
    gpf.utilities.print_summary(model)
    Xnew = np.linspace(0, 3, 100).reshape([100, 1])
    y_dist = model.predict_y(Xnew)
    ymu = y_dist.mean()
    import matplotlib.pyplot as plt
    plt.plot(Xnew, ymu)
    plt.show()
