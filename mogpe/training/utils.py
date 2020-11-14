#!/usr/bin/env python3
import gpflow as gpf
import numpy as np
import pathlib
import tensorflow as tf


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


def save_param_dict(model, log_dir):
    save_model_dir = log_dir + "-param_dict.pickle"
    param_dict = gpf.utilities.parameter_dict(model)
    # import json
    # json = json.dumps(param_dict)
    # f = open(save_model_dir, "w")
    # f.write(json)
    # f.close()
    import pickle
    f = open(save_model_dir, "wb")
    pickle.dump(param_dict, f)
    f.close()


if __name__ == "__main__":
    load_dir = '/Users/aidanscannell/Developer/python-projects/mixture-of-k-gp-experts/models/logs/quadcopter/09-02-151153-param_dict.npz'
    load_model_params(load_dir)
