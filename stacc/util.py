import os
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pooch
import torch
from imageio.v3 import imread

from .unet_2d import UNet2d


def get_postprocessing_parameters(model_name: str) -> Tuple[float, float]:
    """Get the best postprocessing parameters for a counting model.

    These parameters are used for `skimage.features.peak_local_max` to find
    maxima in the output predicted by the network.

    Args:
        model_name: The name of the model. The models 'cells' and 'colonies' are supported.

    Returns:
        The value for min_distance, the minimal distance between colonies.
        The value for threshold_abs, the output intensity threshold.
    """
    assert model_name in ("cells", "colonies")
    if model_name == "cells":
        min_distance = 2
        threshold_abs = 1.0
    else:
        min_distance = 15
        threshold_abs = 2.4
    return min_distance, threshold_abs


def _get_model_registry():
    registry = {
        "cells": "1f62a48d25e86a461777f282ff41a534655c03ecc47e3b61f2f67f117583ac94",
        "colonies": "c8380efff2725bd9d2c464a09b422a0456c736509bc4e280d3e91d8ba4f38cc7",
    }
    urls = {
        "cells": "https://owncloud.gwdg.de/index.php/s/IG9e8N1AuBk7vwL/download",
        "colonies": "https://owncloud.gwdg.de/index.php/s/L9xTn937t8YpIzD/download",
    }
    cache_dir = os.path.expanduser(pooch.os_cache("stacc"))
    models = pooch.create(
        path=os.path.join(cache_dir, "models"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    return models


def get_model_path(model_name: str) -> str:
    """Return the filepath to a pretrained model.

    Args:
        model_name: The name of the model. The models 'cells' and 'colonies' are supported.

    Returns:
        The path to the saved model weights.
    """
    if model_name not in ("cells", "colonies"):
        raise ValueError(f"Invalid model name, expected one of 'cells', 'colonies', got {model_name}.")
    model_registry = _get_model_registry()
    model_path = model_registry.fetch(model_name)
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Could not find the model for counting {model_name} at {model_path}. "
            "This is likely because you have installed the package in an incorrect way. "
            "Please follow the installation instructions in the documentation exactly and try again."
        )
    return model_path


def get_model(model_name: str, model_path: Optional[Union[str, Path]] = None) -> UNet2d:
    """Get the model for colony counting.

    Args:
        model_name: The name of the model. The models 'cells' and 'colonies' are supported.
        model_path: Optional path to a pretrained model. If not given, the default cell / colony model will be loaded.

    Returns:
        The model.
    """
    if model_path is None:
        model_path = get_model_path(model_name)

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
    """@private
    """
    raw = raw.astype("float32")

    mean = raw.mean(axis=axis, keepdims=True) if mean is None else mean
    raw -= mean

    std = raw.std(axis=axis, keepdims=True) if std is None else std
    raw /= (std + eps)

    return raw


def _get_default_device():
    """Copied from MicroSAM
    """
    # check that we're in CI and use the CPU if we are
    # otherwise the tests may run out of memory on MAC if MPS is used.
    if os.getenv("GITHUB_ACTIONS") == "true":
        return "cpu"
    # Use cuda enabled gpu if it's available.
    if torch.cuda.is_available():
        device = "cuda"
    # As second priority use mps.
    # See https://pytorch.org/docs/stable/notes/mps.html for details
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using apple MPS device.")
        device = "mps"
    # Use the CPU as fallback.
    else:
        device = "cpu"
    return device


def get_device(device: Optional[Union[str, torch.device]] = None) -> Union[str, torch.device]:
    """Get the torch device.

    If no device is passed the default device for your system is used.
    Else it will be checked if the device you have passed is supported.

    Args:
        device: The input device.

    Returns:
        The device.
    """
    if device is None or device == "auto":
        device = _get_default_device()
    else:
        device_type = device if isinstance(device, str) else device.type
        if device_type.lower() == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA backend is not available.")
        elif device_type.lower() == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise RuntimeError("PyTorch MPS backend is not available or is not built correctly.")
        elif device_type.lower() == "cpu":
            pass  # cpu is always available
        else:
            raise RuntimeError(f"Unsupported device: {device}\n"
                               "Please choose from 'cpu', 'cuda', or 'mps'.")
    return device


def get_in_channels(image_path):
    # Load the first image to determine the number of channels
    image = np.asarray(imread(image_path))

    # Check if the first image is grayscale or RGB
    if len(image.shape) == 2:
        in_channels = 1
        # print(f"About to process grayscale images")
    elif image.shape[-1] == 4:
        in_channels = 3
        # print(f"About to process RGB images")
    else:
        in_channels = image.shape[-1]
        # print(f"About to process images of dimensions = {image.shape}")

    return in_channels
