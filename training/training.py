import torch
from PIL import ImageFile
from colony_utils import TrainUNET2D
from .utils import StaccDataLoader
from torch.nn import MSELoss
from torch_em.model import UNet2d
from colony_utils.utils import get_in_channels
ImageFile.LOAD_TRUNCATED_IMAGES = True


# TODO def hier Default Stacc Dataloader


def stacc_training(model_name: str,
                   train_images: list, 
                   train_labels: list, 
                   val_images: list, 
                   val_labels: list, 
                   test_images: list, 
                   test_labels: list,
                   patch_shape: tuple, 
                   num_workers: int, 
                   is_pretrained: bool, 
                   checkpoint_path, 
                   batch_size, 
                   iterations, 
                   learning_rate=1e-4, 
                   sigma=None, 
                   lower_bound=None, 
                   upper_bound=None
                   ):
    """

    """
    

    # TODO add patch shape // 16 true

    train_loader, val_loader, _ = StaccDataLoader(train_images, train_labels, val_images, val_labels, test_images, test_labels, 
                                                    patch_shape=patch_shape, num_workers=num_workers, batch_size=batch_size, 
                                                    sigma=sigma, lower_bound=lower_bound, upper_bound=upper_bound)
    
    in_channels = get_in_channels(train_images[0])
    model = UNet2d(in_channels=in_channels, out_channels=1)

    if is_pretrained:    
        device = torch.cuda.current_device()
        model_state = torch.load(checkpoint_path, map_location=torch.device(device))['model_state']
        model.load_state_dict(model_state) 
    
    print(f"Start Training: {model_name}.")

    TrainUNET2D(model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_function=MSELoss,
                learning_rate=learning_rate,
                iterations=iterations,
                device=torch.device("cuda"),
                save_root = "./experiments")
    print(f"Training done.")
