import argparse
import json
import torch
from PIL import ImageFile
from colony_utils import split_dict_dataset
from colony_utils import TrainUNET2D
from colony_utils import CreateDataLoader
from colony_utils import PRLoss
from torch_em.model import UNet2d
from colony_utils.utils import get_in_channels
# change imports to utils
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    config_path = args.config

    # Read config file
    with open(config_path) as file:
        parameters_dict = json.load(file)

    # Get parameters from config file
    # festlegen?
    patch_shape = tuple(parameters_dict["patch_shape"])
    batch_size = parameters_dict["batch_size"]
    num_workers = int(parameters_dict["num_workers"])
    my_iterations = parameters_dict["my_iterations"]
    alpha_loss = parameters_dict["alpha_loss"]
    # dieser path ist ja sehr individuell, müsste required sein
    checkpoint_path = parameters_dict["checkpoints"]
    model_name = parameters_dict["model_name"]
    learning_rate = parameters_dict["learning_rate"]
    pretrained = True if parameters_dict["pretrained"] == 1 else False
    json_dataset = parameters_dict["dataset"]
    epsilon = parameters_dict["eps"]
    sigma = parameters_dict["sigma"]
    lower_bound = parameters_dict["lower_bound"]
    upper_bound = parameters_dict["upper_bound"]
    
    # Read data path file
    with open(json_dataset) as dataset:
        dict_dataset = json.load(dataset)

    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_dict_dataset(dict_dataset)
    
    train_loader, val_loader, _ = CreateDataLoader(train_images, train_labels, val_images, val_labels, test_images, test_labels, 
                                                    patch_shape=patch_shape, num_workers=num_workers, batch_size=batch_size, 
                                                    eps=epsilon, sigma=sigma, lower_bound=lower_bound, upper_bound=upper_bound)

    in_channels = get_in_channels(train_images[0])
    model = UNet2d(in_channels=in_channels, out_channels=1)
    if pretrained:    
        device = torch.cuda.current_device()
        state = torch.load(checkpoint_path, map_location=torch.device(device))['model_state']
        model.load_state_dict(state) 
    
    print(f"Start Training: {model_name}.")
    # PR LOSS is just L2 loss... rewrite for clarity
    TrainUNET2D(model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_function=PRLoss(alpha=alpha_loss),
                learning_rate=learning_rate,
                iterations=my_iterations,
                device=torch.device("cuda"),
                save_root = "/scratch-emmy/usr/nimjjere/models")
    print(f"Training done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args() 
    main(args)