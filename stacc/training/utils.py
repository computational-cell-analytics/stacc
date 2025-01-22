import torch
import os
import json
from typing import Optional, Union, Tuple, List
import numpy as np
from imageio.v3 import imread
from torch.utils.data import DataLoader
from dataclasses import dataclass

# stacc package imports
from training import StaccImageCollectionDataset


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

def _get_device(device: Optional[Union[str, torch.device]] = None) -> Union[str, torch.device]:
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


def _split_data_paths_into_training_dataset(dataset_file):
    with open(dataset_file) as dataset:
        dict_dataset = json.load(dataset)

    train_images = dict_dataset['train']['images']
    train_labels = dict_dataset['train']['labels']

    val_images = dict_dataset['val']['images']
    val_labels = dict_dataset['val']['labels']

    test_images = dict_dataset['test']['images']
    test_labels = dict_dataset['test']['labels']

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def StaccDataLoader(train_dataset_file, patch_shape, num_workers, batch_size, eps=None, sigma=None, lower_bound=None, upper_bound=None):
    
    train_images, train_labels, val_images, val_labels, test_images, test_labels = _split_data_paths_into_training_dataset(train_dataset_file)

    train_set = StaccImageCollectionDataset(train_images, train_labels, patch_shape, eps=eps, sigma=sigma, 
                                                 lower_bound=lower_bound, upper_bound=upper_bound)
    val_set = StaccImageCollectionDataset(val_images, val_labels, patch_shape, eps=eps, sigma=sigma, 
                                               lower_bound=lower_bound, upper_bound=upper_bound)
    test_set = StaccImageCollectionDataset(test_images, test_labels, patch_shape, eps=eps, sigma=sigma, 
                                                lower_bound=lower_bound, upper_bound=upper_bound)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)

    train_dataloader.shuffle = True
    val_dataloader.shuffle = True
    test_dataloader.shuffle = True

    return train_dataloader, val_dataloader, test_dataloader

@dataclass
class TrainingConfig:
    """
    Data class to store training configuration parameters.

    Attributes:
        model_name (str): The name of the model.
        train_dataset (str): Path to the training dataset file. Should be a json, containing a dictionary with train, val, and test image and corresponding label paths.
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