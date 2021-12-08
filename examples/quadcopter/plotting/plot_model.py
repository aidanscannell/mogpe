#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from config import config_from_toml
from mogpe.helpers.quadcopter_plotter import QuadcopterPlotter
from mogpe.training import load_model_from_config_and_checkpoint
from quadcopter.data.load_data import load_quadcopter_dataset

if __name__ == "__main__":
    # ckpt_dir = "./quadcopter/saved_ckpts/subset/2_experts/batch_size_64/learning_rate_0.01/09-29-094538"
    # config_file = "./quadcopter/configs/config_2_experts_subset.toml"  # path to config
    # save_dir = "./quadcopter/images/subset-yo"

    # # ckpt_dir = "./quadcopter/saved_ckpts/subset-10/2_experts/batch_size_64/learning_rate_0.01/further_bound/10-11-104617"
    ckpt_dir = "./quadcopter/saved_ckpts/subset-10/2_experts/batch_size_64/learning_rate_0.01/further_gating_bound/num_inducing_100/11-04-210440"
    save_dir = "./quadcopter/images/subset-10"

    # ckpt_dir = "./logs/quadcopter/subset-10/2_experts/batch_size_64/learning_rate_0.01/further_gating_bound/num_inducing_100/11-04-152145"
    # config_file = "./quadcopter/configs/config_2_experts_subset_10_icra.toml"
    # save_dir = "./quadcopter/images/subset-10-icra"

    ckpt_dir = "./logs/quadcopter/subset-10/2_experts/batch_size_64/learning_rate_0.01/further_gating_bound/num_inducing_100/11-05-104542"
    save_dir = "./quadcopter/images/subset-10"

    # Load config (with model and training params) from toml file
    try:
        if config_file is None:
            config_file = ckpt_dir + "/config.toml"
    except:
        config_file = ckpt_dir + "/config.toml"
    cfg = config_from_toml(config_file, read_from_file=True)

    # Load quadcopter data set
    if cfg.trim_coords is not None:
        trim_coords = np.array(cfg.trim_coords)
    else:
        trim_coords = None
    dataset, _ = load_quadcopter_dataset(
        cfg.data_file,
        trim_coords=trim_coords,
        num_outputs=2,
        plot=False,
        standardise=False,
    )

    def unstandardise(dataset, standardised_dataset):
        def unstandardise_array(std_array, unstd_array):
            mean = tf.reduce_mean(unstd_array)
            std = tf.sqrt(tf.math.reduce_variance(unstd_array))
            print("std")
            print(std)
            return std_array * std + mean

        X, Y = dataset
        standardised_X, standardised_Y = standardised_dataset
        unstd_X = unstandardise_array(standardised_X, X)
        unstd_Y = unstandardise_array(standardised_Y, Y)
        return (unstd_X, unstd_Y)

    standardised_dataset, _ = load_quadcopter_dataset(
        cfg.data_file,
        trim_coords=trim_coords,
        num_outputs=2,
        plot=False,
        standardise=True,
    )
    dataset = unstandardise(dataset, standardised_dataset)
    model = load_model_from_config_and_checkpoint(config_file, ckpt_dir, dataset)

    plotter = QuadcopterPlotter(model, X=dataset[0], Y=dataset[1])
    # plotter.plot_experts_f("./quadcopter/images/subset/test_new.pdf")
    # plotter.plot_model("./quadcopter/images/subset")
    plotter.plot_model(save_dir)
