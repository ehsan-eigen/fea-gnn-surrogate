import json
import numpy as np


def load_config(path, mode="train"):
    with open(path, "r") as f:
        conf = json.load(f)[mode]

    conf["possible_columns_up"] = np.array(conf["possible_columns_up"])
    conf["possible_columns_down"] = np.array(conf["possible_columns_down"])
    return conf
