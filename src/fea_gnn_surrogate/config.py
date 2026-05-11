import json
import numpy as np


DEFAULT_TRAIN_DATASETS = ["dataset_1", "dataset_2", "dataset_6"]


def load_config(path, dataset):
    with open(path, "r") as f:
        conf = json.load(f)[dataset]

    conf["possible_columns_up"] = np.array(conf["possible_columns_up"])
    conf["possible_columns_down"] = np.array(conf["possible_columns_down"])
    return conf


def load_configs(path, datasets):
    with open(path, "r") as f:
        raw = json.load(f)

    configs = []
    for name in datasets:
        conf = raw[name]
        conf["possible_columns_up"] = np.array(conf["possible_columns_up"])
        conf["possible_columns_down"] = np.array(conf["possible_columns_down"])
        configs.append(conf)
    return configs
