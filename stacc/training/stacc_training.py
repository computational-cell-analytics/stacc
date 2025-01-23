import torch
import json
import os
import warnings
import torch_em
from torch_em.model import UNet2d
from contextlib import contextmanager, nullcontext
from typing import Optional, Union

# stacc package imports
from stacc import get_device, StaccDataLoader


def _check_loaders(train_loader, val_loader):
    """
    Helper function to validate the data loaders for training and validation.

    This function checks the following for 2D images:
    - Ensures that the number of channels in the images is either 1 (grayscale) or 3 (RGB).
    - Confirms that the number of channels is consistent between training and validation datasets.
    - Verifies that the dimensions of the images (height and width) are divisible by 16, which is often required for certain neural network architectures.

    Args:
        train_loader (DataLoader): The data loader for the training dataset.
        val_loader (DataLoader): The data loader for the validation dataset.

    Returns:
        int: The number of channels in the images.

    Raises:
        ValueError: If the number of channels is not 1 or 3, if there is a mismatch in the number of channels between training and validation images, or if the image dimensions are not divisible by 16.
    """
    x_train, _ = next(iter(train_loader))  # example image train
    x_val, _ = next(iter(val_loader))      # example image val

    n_channels_train = x_train.shape[1]
    n_channels_val = x_val.shape[1]

    # Ensure the images are 2D
    if len(x_train.shape) != 4 or len(x_val.shape) != 4:
        raise ValueError(
            "Expected 2D images with shape (batch_size, channels, height, width). "
            f"Got shapes: {x_train.shape} and {x_val.shape}."
        )

    # Check if the dimensions of the images are divisible by 16
    if any(dim % 16 != 0 for dim in x_train.shape[2:]):
        raise ValueError(
            "Training image dimensions (height and width) must be divisible by 16. "
            f"Got dimensions: {x_train.shape[2:]}."
        )
    
    if any(dim % 16 != 0 for dim in x_val.shape[2:]):
        raise ValueError(
            "Validation image dimensions (height and width) must be divisible by 16. "
            f"Got dimensions: {x_val.shape[2:]}."
        )

    # Check for grayscale or RGB
    if n_channels_train not in (1, 3) or n_channels_val not in (1, 3):
        raise ValueError(
            "Invalid number of channels for the input data from the data loader. "
            f"Expect 1 or 3 channels, got {n_channels_train} and {n_channels_val}."
        )
    
    if n_channels_train != n_channels_val:
        raise ValueError(
            "Mismatch in number of channels in training and validation images. "
            f"Got {n_channels_train} in the training loader and {n_channels_val} in the validation loader."
        )

    return n_channels_train


@contextmanager
def _filter_warnings(ignore_warnings):
    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    else:
        with nullcontext():
            yield


def run_stacc_training(
        model_name: str,
        train_loader: StaccDataLoader,
        val_loader: StaccDataLoader,
        n_epochs: int = 25,
        learning_rate: Optional[float] = 1e-4, 
        device: Optional[Union[str, torch.device]] = None,
        pretrained_model_path: Optional[Union[str, os.PathLike]] = None, 
        save_new_model_path: Optional[Union[str, os.PathLike]] = None,
        iterations: Optional[int] = None, 
    ) -> None:
    """
    Run training for the STACC model.

    This function sets up and runs the training process for a STACC model using the provided data loaders.

    Args:
        model_name (str): The name of the model.
        train_loader (StaccDataLoader): Data loader for the training dataset.
        val_loader (StaccDataLoader): Data loader for the validation dataset.
        n_epochs (int, optional): Number of epochs for training. Defaults to 25.
        learning_rate (Optional[float], optional): Learning rate for the optimizer. Defaults to 1e-4.
        device (Optional[Union[str, torch.device]], optional): Device to run the training on. Defaults to None.
        pretrained_model_path (Optional[Union[str, os.PathLike]], optional): Path to a pretrained model checkpoint. Defaults to None.
        save_new_model_path (Optional[Union[str, os.PathLike]], optional): Path to save the new model. Defaults to None.
        iterations (Optional[int], optional): Number of iterations for training. Defaults to None.

    Returns:
        None
    """

    with _filter_warnings(ignore_warnings=True):
        n_input_channels = _check_loaders(train_loader, val_loader)
        
        model = UNet2d(in_channels=n_input_channels, out_channels=1)
        device = get_device(device)

        trainer = torch_em.default_segmentation_trainer(
            name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss=torch.nn.MSELoss(),
            metric=torch.nn.MSELoss(),
            learning_rate=learning_rate,
            device=device,
            mixed_precision=True,
            log_image_interval=100,
            save_root=save_new_model_path,
            compile_model=False,
            logger=None
        )
        
        trainer_fit_params = {"epochs": n_epochs} if iterations is None else {"iterations": iterations}
        trainer.fit(**trainer_fit_params, load_from_checkpoint=pretrained_model_path)





