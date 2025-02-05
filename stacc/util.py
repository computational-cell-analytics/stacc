import os
import torch
import os
import json
from typing import Optional, Union, Tuple, List
import numpy as np
from imageio.v3 import imread
from dataclasses import dataclass
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

def get_model(model_name: str) -> UNet2d:
    """Get the model for colony counting.

    Args:
        model_name: The name of the model. The models 'cells' and 'colonies' are supported.

    Returns:
        The model.
    """
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
    """Copied from and modidied based on MicroSAM: Get the torch device.

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

@dataclass
class TrainingConfig:
    """
    Data class to store training configuration parameters.

    Attributes:
        model_name (str): The name of the model.
        train_dataset (str): Path to the training dataset file. Should be a json, containing a dictionary with train, val, and test image and corresponding label PATHS.
        pretrained_model_path (str): Path to the pretrained model checkpoint.
        save_new_model_path (str): Path to save the new model checkpoints. By default, latest and best will be stored.
        batch_size (int): Batch size for the data loader.
        patch_shape (Tuple[int, int]): Shape of the patches for training.
        n_workers (int): Number of workers for data loading.
        iterations (int): Number of iterations for training.
        n_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        epsilon (float): Truncate value for Gaussian stamp in STACC.
        sigma (float): Sigma parameter for STACC. Determines the size of the Gaussian stamp.
        lower_bound (float): Lower bound for STACC. Lower bound for Gaussian stamp size.
        upper_bound (float): Upper bound for STACC. Upper bound for Gaussian stamp size.
        augmentations (List[str]): List of augmentations to apply.
    """
    model_name: str
    train_dataset: str
    test_dataset: str
    pretrained_model_path: str
    save_new_model_path: str
    batch_size: int
    patch_shape: Tuple[int, int]
    n_workers: int
    iterations: int
    n_epochs: int
    learning_rate: float
    epsilon: float
    sigma: float
    lower_bound: float
    upper_bound: float
    augmentations: List[str]
    comment: str

def load_config(config_file_path: str) -> TrainingConfig:
    """
    Load training configuration from a JSON file.

    Args:
        config_file_path (str): Path to the JSON configuration file.

    Returns:
        TrainingConfig: An instance of TrainingConfig with parameters loaded from the file.
    """
    with open(config_file_path) as file:
        parameters_dict = json.load(file)
    return TrainingConfig(**parameters_dict)

