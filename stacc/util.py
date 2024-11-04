import os
import torch

from .unet_2d import UNet2d


def get_postprocessing_parameters(model_name):
    assert model_name in ("cells", "colonies")
    if model_name == "cells":
        min_distance = 2
        threshold_abs = 1.0
    else:
        min_distance = 15
        threshold_abs = 2.4
    return min_distance, threshold_abs


def get_model(model_name):
    assert model_name in ("cells", "colonies")
    fpath = os.path.split(__file__)[0]
    model_path = os.path.join(fpath, "..", "models", f"{model_name}.pt")

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Could not find the model for counting {model_name}."
            "This is likely because you have installed the package in an incorrect way."
            "Please follow the installation instructions in the README exactly and try again."
        )

    if model_name == "cells":
        model_kwargs = {
            "in_channels": 1, "out_channels": 1, "depth": 4,
            "initial_features": 32, "gain": 2, "final_activation": None
        }
    else:
        model_kwargs = {
            "in_channels": 3, "out_channels": 1, "depth": 4,
            "initial_features": 32, "gain": 2, "final_activation": None
        }

    model_state = torch.load(model_path, weights_only=True)
    model = UNet2d(**model_kwargs)
    model.load_state_dict(model_state)

    return model


def standardize(raw, mean=None, std=None, axis=None, eps=1e-7):
    raw = raw.astype("float32")

    mean = raw.mean(axis=axis, keepdims=True) if mean is None else mean
    raw -= mean

    std = raw.std(axis=axis, keepdims=True) if std is None else std
    raw /= (std + eps)

    return raw
