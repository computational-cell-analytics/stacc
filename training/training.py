import torch
import os
import warnings
import contextmanager, nullcontext
import torch_em
from typing import Optional, Union
from .utils import StaccDataLoader
from .utils import get_device


# TODO is this still necessary?
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



# TODO
def _check_loader():
    return

# TODO Constantin: We want to ignore warning and only print erros, right?
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
        num_input_channels: int,
        train_loader: StaccDataLoader,
        val_loader: StaccDataLoader,
        epochs: int = 25,
        device: Optional[Union[str, torch.device]] = None,
        is_pretrained: Optional[bool] = False, 
        pretrained_model_path: Optional[Union[str, os.PathLike]] = None, 
        save_new_model_path: Optional[Union[str, os.PathLike]] = None, # TODO Constantin: Hier None oder sowas wie ./experiments?
        iterations: Optional[int] = None, 
        learning_rate: Optional[float] = 1e-4, 
    ) -> None: # TODO Constantin: -> None macht doch einfach gar nichts, oder?
    
    """ Run training for STACC model.

    Args:
    TODO
    
    """
    with _filter_warnings():

        _check_loader(train_loader)
        _check_loader(val_loader)
        
        model = torch_em.UNet2d(in_channels=num_input_channels, out_channels=1)
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
            trainer_fit_params = {"epochs": epochs}
        else:
            trainer_fit_params = {"iterations": iterations}

        trainer.fit(**trainer_fit_params, load_from_checkpoint=pretrained_model_path)

