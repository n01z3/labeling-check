import os
import yaml


def get_path(path="../config/paths.yml"):
    with open(path, 'r') as stream:
        config = yaml.load(stream)
    return config
