import torch
import os
import warnings
import contextmanager, nullcontext
import torch_em
from typing import Optional, Union
from .utils import StaccDataLoader
from .utils import get_device

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _check_loaders(train_loader, val_loader):
    x_train, _ = next(iter(train_loader)) # example image train
    x_val, _ = next(iter(val_loader)) # example image val
 
    n_channels_train = x_train.shape[1]
    n_channels_val = x_val.shape[1]

    # grayscale or RGB
    if n_channels_train not in (1, 3) or n_channels_val not in (1, 3):
        raise ValueError(
            "Invalid number of channels for the input data from the data loader."
            f"Expect 1 or 3 channels, got {n_channels_train} and {n_channels_val}."
        )
    
    if n_channels_train != n_channels_val:
        raise ValueError(
            "Mismatch in number of channels in training and validation images."
            f"Got {n_channels_train} in the training loader 
            and {n_channels_val} in the validation loader."
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

def stacc_training(
        model_name: str,
        train_loader: StaccDataLoader,
        val_loader: StaccDataLoader,
        n_epochs: int = 25,
        device: Optional[Union[str, torch.device]] = None,
        pretrained_model_path: Optional[Union[str, os.PathLike]] = None, 
        save_new_model_path: Optional[Union[str, os.PathLike]] = None,
        iterations: Optional[int] = None, 
        learning_rate: Optional[float] = 1e-4, 
    ) -> None:
    
    """ Run training for STACC model.

    Args:
    TODO
    
    """
    with _filter_warnings():

        n_input_channels = _check_loaders(train_loader, val_loader)
        
        model = torch_em.UNet2d(in_channels=n_input_channels, out_channels=1)
        device = get_device(device)

        trainer = torch_em.default_segmentation_trainer(
                name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss=torch.nn.MSELoss,
                metric=torch.nn.MSELoss,
                learning_rate=learning_rate,
                device=device,
                mixed_precision=True,
                log_image_interval=100,
                save_root = save_new_model_path,
                compile_model= False,
                logger=None
                )
        
        if iterations is None:
            trainer_fit_params = {"epochs": n_epochs}
        else:
            trainer_fit_params = {"iterations": iterations}

        trainer.fit(**trainer_fit_params, load_from_checkpoint=pretrained_model_path)

