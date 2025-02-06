import json
from dataclasses import dataclass
from typing import Tuple, List

import torch


def export_model(checkpoint_path: str, export_path: str) -> None:
    """Export a trained model from a checkpoint.

    The exported model can then be used within the napari plugin or the CLI for counting.

    Args:
        checkpoint_path: The path to the checkpoint.
        export_path: Where to save the model.
    """
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = model["model_state"]
    model_kwargs = model["init"]["model_kwargs"]
    print("The model from", checkpoint_path, "was created with the following kwargs:")
    print(model_kwargs)
    torch.save(model_state, export_path)


@dataclass
class TrainingConfig:
    """Data class to store training configuration parameters.

    Attributes:
        model_name: The name of the model.
        train_dataset: Path to the training dataset file.
            Should be a json, containing a dictionary with train, val, and test image and corresponding label paths.
        pretrained_model_path: Path to the pretrained model checkpoint.
        save_new_model_path: Path to save the new model checkpoints. By default, latest and best will be stored.
        batch_size: Batch size for the data loader.
        patch_shape: Shape of the patches for training.
        n_workers: Number of workers for data loading.
        iterations: Number of iterations for training.
        n_epochs: Number of epochs for training.
        learning_rate: Learning rate for the optimizer.
        epsilon: Truncate value for Gaussian stamp in STACC.
        sigma: Sigma parameter for STACC. Determines the size of the Gaussian stamp.
        lower_bound: Lower bound for STACC. Lower bound for Gaussian stamp size.
        upper_bound: Upper bound for STACC. Upper bound for Gaussian stamp size.
        augmentations: List of augmentations to apply.
        comment: Additional training description
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
    """Load training configuration from a JSON file.

    Args:
        config_file_path: Path to the JSON configuration file.

    Returns:
        An instance of TrainingConfig with parameters loaded from the file.
    """
    with open(config_file_path) as file:
        parameters_dict = json.load(file)
    return TrainingConfig(**parameters_dict)
