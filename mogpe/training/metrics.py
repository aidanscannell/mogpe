#!/usr/bin/env python3
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def negative_log_predictive_density(model, dataset):
    y_dist = model.predict_y(dataset[0])
    if isinstance(y_dist, tfd.Distribution):
        log_probs = model.predict_y(dataset[0]).log_prob(dataset[1])
    else:
        log_probs = tfd.Normal(y_dist[0], y_dist[1]).log_prob(dataset[1])
    nlpd = -tf.reduce_mean(log_probs)
    # nlpd = -tf.reduce_sum(log_probs)
    return nlpd


def build_negative_log_predictive_density(model, dataset, compile: bool = True):
    def closure():
        batch = next(dataset)
        return negative_log_predictive_density(model=model, dataset=batch)

    if compile:
        return tf.function(closure)
    else:
        return closure


def mean_absolute_error(model, dataset):
    y_dist = model.predict_y(dataset[0])
    if isinstance(y_dist, tfd.Distribution):
        y_mean = y_dist.mean()
    else:
        y_mean = y_dist[0]
    error = tf.reduce_mean(tf.math.abs(y_mean - dataset[1]))
    return error


def build_mean_absolute_error(model, dataset, compile: bool = True):
    def closure():
        batch = next(dataset)
        return mean_absolute_error(model=model, dataset=batch)

    if compile:
        return tf.function(closure)
    else:
        return closure


def root_mean_squared_error(model, dataset, batched=False):
    """
    dataset: tuple of input-output pairs (X, Y)
        X tensor with shape [N, D] or [Batch, N, D]
        Y tensor with shape [N, F] or [Batch, N, F]
    returns: RMSE tensor [] or [Batch]
    """
    X, Y = dataset
    y_dist = model.predict_y(X)
    if isinstance(y_dist, tfd.Distribution):
        y_mean = y_dist.mean()
    else:
        y_mean = y_dist[0]
    if batched:
        y_mean = tf.expand_dims(y_mean, -2)
        Y = tf.expand_dims(Y, -2)
    error = (y_mean - Y) ** 2
    mean_error = tf.reduce_mean(error, [-1, -2])
    rmse = tf.math.sqrt(mean_error)
    return rmse


def build_root_mean_squared_error(model, dataset, compile: bool = True):
    def closure():
        batch = next(dataset)
        return root_mean_squared_error(model=model, dataset=batch)

    if compile:
        return tf.function(closure)
    else:
        return closure
