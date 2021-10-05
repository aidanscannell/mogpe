#!/usr/bin/env python3
import numpy as np
from config import config_from_toml
from mogpe.helpers.quadcopter_plotter import QuadcopterPlotter
from mogpe.training import load_model_from_config_and_checkpoint
from quadcopter.load_data import load_quadcopter_dataset

if __name__ == "__main__":
    ckpt_dir = "./quadcopter/saved_ckpts/subset/2_experts/batch_size_64/learning_rate_0.01/09-29-094538"

    # Load config (with model and training params) from toml file
    config_file = "./quadcopter/configs/config_2_experts_subset.toml"  # path to config
    cfg = config_from_toml(config_file, read_from_file=True)

    # Load quadcopter data set
    if cfg.trim_coords is not None:
        trim_coords = np.array(cfg.trim_coords)
    else:
        trim_coords = None
    dataset = load_quadcopter_dataset(
        cfg.data_file,
        trim_coords=trim_coords,
        num_outputs=2,
        plot=False,
        standardise=False,
    )
    model = load_model_from_config_and_checkpoint(config_file, ckpt_dir, dataset)

    plotter = QuadcopterPlotter(model, X=dataset[0], Y=dataset[1])
    # plotter.plot_experts_f("./quadcopter/images/subset/test_new.pdf")
    plotter.plot_model("./quadcopter/images/subset")
