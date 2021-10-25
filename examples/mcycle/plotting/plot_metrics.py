#!/usr/bin/env python3

import gpflow as gpf
import tensorflow as tf
from config import config_from_toml
from mcycle.data.load_data import load_mcycle_dataset
from mcycle.train_gp_on_mcycle_from_config import init_gp_from_config
from mcycle.train_svgp_on_mcycle_from_config import init_svgp_from_config
from mogpe.training import load_model_from_config_and_checkpoint
from mogpe.training.metrics import (
    mean_absolute_error,
    negative_log_predictive_density,
    root_mean_squared_error,
)


def restore_model_from_ckpt(model, ckpt_dir):
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)
    print("Restored Model")
    gpf.utilities.print_summary(model)
    return model


def restore_svgp(config_file, ckpt_dir):
    """Restore SVGP checkpoint"""
    svgp_model = init_svgp_from_config(config_file)
    return restore_model_from_ckpt(svgp_model, ckpt_dir)


def restore_gp(config_file, ckpt_dir):
    """Restore GP checkpoint"""
    gp_model = init_gp_from_config(config_file)
    return restore_model_from_ckpt(gp_model, ckpt_dir)


if __name__ == "__main__":
    ckpt_dirs = {
        "K=2, L1": "./mcycle/saved_ckpts/2_experts/batch_size_16/learning_rate_0.01/tight_bound/num_inducing_32/10-14-172144",
        "K=2, L2": "./mcycle/saved_ckpts/2_experts/batch_size_16/learning_rate_0.01/further_bound/num_inducing_32/10-14-172119",
        "K=3, L1": "./mcycle/saved_ckpts/3_experts/batch_size_16/learning_rate_0.01/tight_bound/num_inducing_32/10-14-172452",
        "K=3, L2": "./mcycle/saved_ckpts/3_experts/batch_size_16/learning_rate_0.01/further_bound/num_inducing_32/10-14-172350",
        "SVGP, M=16": "./mcycle/saved_ckpts/svgp/1_experts/batch_size_16/learning_rate_0.01/normal_bound/num_inducing_16/10-15-095606",
        "SVGP, M=32": "./mcycle/saved_ckpts/svgp/1_experts/batch_size_16/learning_rate_0.01/normal_bound/num_inducing_32/10-15-095647",
        "GP": "./mcycle/saved_ckpts/gp/1_experts/batch_size_na/learning_rate_0.001/None_bound/10-14-175702",
    }
    configs = {
        "K=2, L1": "./mcycle/configs/config_2_experts.toml",
        "K=2, L2": "./mcycle/configs/config_2_experts.toml",
        "K=3, L1": "./mcycle/configs/config_3_experts.toml",
        "K=3, L2": "./mcycle/configs/config_3_experts.toml",
        "SVGP, M=16": "./mcycle/configs/config_svgp_m_16.toml",
        "SVGP, M=32": "./mcycle/configs/config_svgp_m_32.toml",
        "GP": "./mcycle/configs/config_gp.toml",
    }

    results = {}
    for model_str in ckpt_dirs:
        cfg = config_from_toml(configs[model_str], read_from_file=True)

        # Load mcycle data set
        train_dataset, test_dataset = load_mcycle_dataset(
            cfg.data_file,
            plot=False,
            standardise=cfg.standardise,
            test_split_size=cfg.test_split_size,
        )

        # Restore MoSVGPE checkpoint
        if "K=" in model_str:
            model = load_model_from_config_and_checkpoint(
                configs[model_str], ckpt_dirs[model_str], train_dataset
            )
        elif "SVGP" in model_str:
            model = restore_svgp(configs[model_str], ckpt_dirs[model_str])
        elif "GP" in model_str:
            model = restore_gp(configs[model_str], ckpt_dirs[model_str])

        # Calculate metrics
        results["NLPD " + model_str] = negative_log_predictive_density(
            model, test_dataset
        ).numpy()
        results["RMSE " + model_str] = root_mean_squared_error(
            model, test_dataset
        ).numpy()
        results["MAE " + model_str] = mean_absolute_error(model, test_dataset).numpy()

    # Print results
    for item in sorted(results.keys()):
        print(item + ": {}".format(results[item]))
