#!/usr/bin/env python3
from mogpe.training import train_from_config_and_dataset
from mcycle.load_data import load_mcycle_dataset


if __name__ == "__main__":
    # Set path to data set csv file
    data_file = "./data/mcycle.csv"

    # Set path to training config
    config_file = "./configs/config_2_experts.toml"
    # config_file = './configs/config_3_experts.toml'

    # Load mcycle data set
    dataset = load_mcycle_dataset(data_file)

    # Parse the toml config file and train
    trained_model = train_from_config_and_dataset(config_file, dataset)
