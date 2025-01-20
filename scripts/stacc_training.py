import argparse
import json
from dataclasses import dataclass
from typing import Tuple, List, Union
from ..training.utils import StaccDataLoader
from ..training.training import run_stacc_training
from PIL import ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def main(args):
    """
    Main function to execute the training process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing the path to the configuration file.
    """
    config = load_config(args.config)

    train_loader, val_loader, _ = StaccDataLoader(
        config.train_dataset, 
        n_workers=config.n_workers, 
        batch_size=config.batch_size, 
        patch_shape=config.patch_shape,
        sigma=config.sigma, 
        lower_bound=config.lower_bound, 
        upper_bound=config.upper_bound
    )

    run_stacc_training(
        config.model_name, 
        train_loader, 
        val_loader, 
        config.n_epochs, 
        config.learning_rate, 
        device="gpu", 
        pretrained_model_path=config.pretrained_model_path, 
        save_new_model_path=config.save_new_model_path
    )

if __name__ == "__main__":
    # Create an argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train a UNet model for a counting task using a configuration file.")

    # Add an argument for the configuration file and parse it
    parser.add_argument(
        "config", 
        type=str, 
        help="Path to the JSON configuration file containing model and training parameters."
    )
    args = parser.parse_args()

    main(args)
