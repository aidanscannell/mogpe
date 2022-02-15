import yaml
import json
import tensorflow as tf


def model_from_yaml(
    yaml_cfg_filename: str,
    custom_objects: dict = None,
    temp_json_file: str = "/tmp/json_config.json",
):
    with open(yaml_cfg_filename, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    with open(temp_json_file, "w") as f:
        json.dump(cfg, f, sort_keys=False)
    with open(temp_json_file, "r") as f:
        model_cfg = f.read()
    return tf.keras.models.model_from_json(model_cfg, custom_objects=custom_objects)
