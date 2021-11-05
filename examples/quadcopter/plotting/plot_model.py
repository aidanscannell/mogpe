#!/usr/bin/env python3
import numpy as np
from config import config_from_toml
from mogpe.helpers.quadcopter_plotter import QuadcopterPlotter
from mogpe.training import load_model_from_config_and_checkpoint
from quadcopter.data.load_data import load_quadcopter_dataset

if __name__ == "__main__":
    ckpt_dir = "./quadcopter/saved_ckpts/subset/2_experts/batch_size_64/learning_rate_0.01/09-29-094538"
    config_file = "./quadcopter/configs/config_2_experts_subset.toml"  # path to config
    save_dir = "./quadcopter/images/subset-yo"

    # # ckpt_dir = "./quadcopter/saved_ckpts/subset-10/2_experts/batch_size_64/learning_rate_0.01/further_bound/10-11-104617"
    ckpt_dir = "./quadcopter/saved_ckpts/subset-10/2_experts/batch_size_64/learning_rate_0.01/further_gating_bound/num_inducing_100/11-04-210440"
    config_file = "./quadcopter/configs/config_2_experts_subset_10.toml"
    save_dir = "./quadcopter/images/subset-10"

    # ckpt_dir = "./logs/quadcopter/subset-10/2_experts/batch_size_64/learning_rate_0.01/further_gating_bound/num_inducing_100/11-04-152145"
    # config_file = "./quadcopter/configs/config_2_experts_subset_10_icra.toml"
    # save_dir = "./quadcopter/images/subset-10-icra"

    # Load config (with model and training params) from toml file
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
    model = load_model_from_config_and_checkpoint(config_file, ckpt_dir, dataset)

    plotter = QuadcopterPlotter(model, X=dataset[0], Y=dataset[1])
    # plotter.plot_experts_f("./quadcopter/images/subset/test_new.pdf")
    # plotter.plot_model("./quadcopter/images/subset")
    plotter.plot_model(save_dir)
