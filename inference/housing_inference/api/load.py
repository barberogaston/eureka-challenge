import json
import os
from functools import cache

from mlem.api import load


@cache
def load_model(models_path: str, usi: str):
    "Load the model."
    return load(os.path.join(models_path, f"{usi}_model"))


@cache
def load_performance(models_path: str, usi: str):
    "Load the model's performance."
    with open(os.path.join(models_path, f"{usi}_summary.json")) as f:
        return json.load(f)
