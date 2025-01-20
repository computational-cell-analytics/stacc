import itertools
import torch
import os
import json
import argparse

import numpy as np
import imageio.v3 as iio

from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.measure import label
from imageio.v3 import imread
from tqdm import tqdm
from torch_em.model import UNet2d
from colony_utils import json_transform_to_matrix
from colony_utils import ImageCollectionDatasetJsonLabels
from torch.utils.data import DataLoader

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
    """Copied from MicroSAM: Get the torch device.

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

def StaccDataLoader(train_images, train_labels, val_images, val_labels, test_images, test_labels, patch_shape, num_workers, batch_size, eps=0.00001, sigma=None, lower_bound=None, upper_bound=None):
    
    train_set = ImageCollectionDatasetJsonLabels(train_images, train_labels, patch_shape, eps=eps, sigma=sigma, 
                                                 lower_bound=lower_bound, upper_bound=upper_bound)
    val_set = ImageCollectionDatasetJsonLabels(val_images, val_labels, patch_shape, eps=eps, sigma=sigma, 
                                               lower_bound=lower_bound, upper_bound=upper_bound)
    test_set = ImageCollectionDatasetJsonLabels(test_images, test_labels, patch_shape, eps=eps, sigma=sigma, 
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


def split_dict_dataset(dict_dataset):
    # lists of images/labels
    train_images = dict_dataset['train']['images']
    train_labels = dict_dataset['train']['labels']

    val_images = dict_dataset['val']['images']
    val_labels = dict_dataset['val']['labels']

    test_images = dict_dataset['test']['images']
    test_labels = dict_dataset['test']['labels']

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def get_center_coordinates(json_path):
    # Read the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # List to store center coordinates
    center_coordinates = []
    
    # Iterate through the labels
    for label in data['labels']:
        x = label['x']
        y = label['y']
        width = label['width']
        height = label['height']
        
        # Calculate the center coordinates
        center_x = int(x + width / 2)
        center_y = int(y + height / 2)
        
        # Append the center coordinates to the list
        center_coordinates.append([center_y, center_x])
    
    # Convert the list to a NumPy array
    center_coordinates = np.array(center_coordinates)
    
    return center_coordinates

def load_model(config, in_channels=3):
    """
    Load the UNet model from the specified checkpoint.

    Args:
        model_path (str): The full path to the model checkpoint.
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The loaded UNet model.
    """
    model_name = config["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print(f"Warning: CUDA is not available. The model will run on CPU, which may take a while.")
    model = UNet2d(in_channels=in_channels, out_channels=1)
    state = torch.load(os.path.join("/scratch-emmy/usr/nimjjere/models/checkpoints/", model_name, "best.pt"), map_location=torch.device(device))['model_state']
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, device

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

def get_distance_threshold_from_gridsearch(data_frame, distances, threshes):
    combine = list(itertools.product(distances, threshes))
    mean_score = 0
    dist = 0
    thresh = 0.0
    # print(data_frame)
    for c in combine:
        mask = (data_frame["Distance"] == c[0]) & (data_frame["Threshold"] == c[1])
        param_kombi_df = data_frame[mask]
        # print(f"New DF: \n {param_kombi_df}")
        meany = param_kombi_df.mean(axis=0).iloc[0]
        # print(f"mean: {meany}, mean_score: {mean_score}, parameters: {c}")
        if mean_score < meany:
            mean_score = meany
            dist = c[0]
            thresh = c[1]

    return dist, thresh

def get_test_data(data): 
    test_dict = data['test']
    return test_dict['images'], test_dict['labels']

def get_train_data(data): 
    train_dict = data['train']
    return train_dict['images'], train_dict['labels']

def get_image_files(folder_path):
    # Define a set of image file extensions
    image_extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    # List comprehension to filter image files and include the full path
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

def predict_image(image_path, model, patch_shape, device):
    from torch_em.transform.raw import normalize
    from torch_em.util.prediction import predict_with_padding
    
    image = np.asarray(imread(image_path))
    # if grayscale image
    # Check if the image is grayscale
    if len(image.shape) == 2:
        # Add a channel dimension to make it (x, y, 1)
        image = image[..., np.newaxis]
    elif image.shape[-1] == 4:
        # If the image has 4 channels (e.g., RGBA), use only the first 3 channels (RGB)
        image = image[..., :3]
    
    # Resize the image to the configs patch_shape
    image_stand = resize(image, patch_shape, anti_aliasing=True, preserve_range=True).astype(image.dtype)
    image_stand = normalize(image)
    torchi = image_stand.transpose((2,0,1))

    # predict
    preds = predict_with_padding(model=model, input_=torchi, device=device, min_divisible=(16,16), with_channels=True)
    preds = preds.squeeze()
    return preds
    
def blurr_to_point(matrix, dist, thresh):
    maxima = peak_local_max(matrix, min_distance = dist, threshold_abs = thresh)
    point = np.zeros(matrix.shape)
    x = maxima.T[0]
    y = maxima.T[1]
    coords = (x, y)
    point[coords] = 1
    
    return label(point), len(maxima)

def save_samples(loader, prefix, n_samples):
    """
    Stores first n_samples image-label-pairs from dataset. Compress original images to save pace.
    """
    i = 0
    for batch in iter(loader):
        if i >= n_samples:
            break
        
        iio.imwrite(f'{prefix}_{i}_image_0.jpg', ensure_array(batch[0][0]).transpose(1, 2, 0).astype(np.uint8))
        iio.imwrite(f'{prefix}_{i}_image_1.jpg', ensure_array(batch[0][1]).transpose(1, 2, 0).astype(np.uint8))
        iio.imwrite(f'{prefix}_{i}_label_0.tif', np.squeeze(ensure_array(batch[1][0]).transpose(1, 2, 0)))
        iio.imwrite(f'{prefix}_{i}_label_1.tif', np.squeeze(ensure_array(batch[1][1]).transpose(1, 2, 0)))

        i += 1

def ensure_array(array, dtype=None):
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    assert isinstance(array, np.ndarray), f"Cannot convert {type(array)} to numpy"
    if dtype is not None:
        array = np.require(array, dtype=dtype)
    return array

def create_stacc_map(path_to_data, name_of_destination_folder, sigma):
    print(f"Type of sigma: {type(sigma)}, sigma: {sigma}")
    all_jsons = [os.path.join(path_to_data, label) for label in os.listdir(path_to_data) if label.endswith(".json")]
    for json in tqdm(all_jsons):
        label = json_transform_to_matrix(json, eps=0.00001, sigma=sigma)
        iio.imwrite(os.path.join(path_to_data, name_of_destination_folder, os.path.basename(json)[:-5] + ".tif"), label)
        # print("next")
        # print(os.path.join(path_to_data, name_of_destination_folder, os.path.basename(json)[:-5] + ".tif"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_data", type=str)
    parser.add_argument("name_of_destination_folder", type=str)
    parser.add_argument("sigma", type=str)

    args = parser.parse_args()
    if args.sigma == "None":
        sigma=None
    else:
        sigma = int(args.sigma)
    create_stacc_map(args.path_to_data, args.name_of_destination_folder, sigma)