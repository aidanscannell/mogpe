#!/usr/bin/env python3
import json
from json import JSONEncoder
from typing import List, Optional

import gpflow as gpf
import numpy as np
import tensorflow as tf
import yaml
from gpflow.inducing_variables import InducingVariables
from mogpe.custom_types import InputData
from mogpe.keras.mixture_of_experts import MixtureOfSVGPExperts


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def save_json_config(obj, filename: str = "config.json"):
    """Save object to .json using get_config()"""
    cfg = tf.keras.utils.serialize_keras_object(obj)
    with open(filename, "w") as f:
        json.dump(cfg, f, cls=NumpyArrayEncoder)


def load_from_json_config(filename: str, custom_objects: dict):
    """Load object from .json using from_config()"""
    with open(filename, "r") as read_file:
        json_cfg = read_file.read()
    return tf.keras.models.model_from_json(json_cfg, custom_objects=custom_objects)


def model_from_yaml(yaml_cfg_filename: str, custom_objects: dict = None):
    with open(yaml_cfg_filename, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        model_cfg = json.dumps(cfg)
    return tf.keras.models.model_from_json(model_cfg, custom_objects=custom_objects)


def sample_mosvgpe_inducing_inputs_from_data(X: InputData, model: MixtureOfSVGPExperts):
    # TODO should inducing inputs only be for active dims or all inputs?
    for expert in model.experts_list:
        sample_inducing_variables_from_data(X, expert.gp.inducing_variable)
    sample_inducing_variables_from_data(
        X,
        model.gating_network.gp.inducing_variable,
        active_dims=model.gating_network.gp.kernel.active_dims,
    )


def sample_inducing_variables_from_data(
    X: InputData,
    inducing_variable: InducingVariables,
    active_dims: Optional[List[int]] = None,
):
    if isinstance(
        inducing_variable, gpf.inducing_variables.SharedIndependentInducingVariables
    ):
        inducing_variable.inducing_variable.Z.assign(
            sample_inducing_inputs_from_data(
                X,
                inducing_variable.inducing_variable.Z.shape[0],
                active_dims=active_dims,
            )
        )
    elif isinstance(
        inducing_variable, gpf.inducing_variables.SeparateIndependentInducingVariables
    ):
        for inducing_var in inducing_variable.inducing_variables:
            Z = sample_mosvgpe_inducing_inputs_from_data(X, inducing_var.Z)
            inducing_var.Z.assign(Z)
    else:
        inducing_variable.Z.assign(
            sample_inducing_inputs_from_data(
                X, inducing_variable.Z.shape[0], active_dims=active_dims
            )
        )


def sample_inducing_inputs_from_data(
    X: InputData, num_inducing: int, active_dims: Optional[List[int]] = None
):
    idx = np.random.choice(range(X.shape[0]), size=num_inducing, replace=False)
    if isinstance(active_dims, slice):
        X = X[..., active_dims]
    elif active_dims is not None:
        X = tf.gather(X, active_dims, axis=-1)
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    return X[idx, ...].reshape(-1, X.shape[1])
