import argparse
import os

# stacc package imports
from training import load_config
from stacc import run_counting



def _image_or_folder(path):
    if os.path.isfile(path):
        return False
    elif os.path.isdir(path):
        return True
    else:
        raise ValueError(f"{path} does not exist or is not a valid file or directory.")    
    
def _is_valid_image_file(path):
    """
    Checks if the given path is a valid image file with extensions jpg, jpeg, tif, tiff, or png.

    Parameters:
    path (str): The file path to check.

    Returns:
    bool: True if the path is a valid image file, False otherwise.

    Raises:
    ValueError: If the path is not a file or does not have a valid image extension.
    """
    valid_extensions = {'.jpg', '.jpeg', '.tif', '.tiff', '.png'}
    
    if os.path.isfile(path):
        _, file_extension = os.path.splitext(path)
        if file_extension.lower() in valid_extensions:
            return True
        else:
            raise ValueError(f"{path} is a file but does not have a valid image extension. Valid are: '.jpg', '.jpeg', '.tif', '.tiff', '.png'.")
    else:
        raise ValueError(f"{path} is not a valid file. Please provide an image file with one of the following extensions: '.jpg', '.jpeg', '.tif', '.tiff', '.png'.")
    
def main(config_path, prediction_path): 
    config = load_config(config_path)
    is_folder = _image_or_folder(prediction_path)
    os.makedirs(store_path, exist_ok=True)

    if not is_folder:
        # check if file is a valid image file
        # predict image with model from config file
        prediction = ""
        # store prediction as what? points ontop of image? list of coordinates? csv file?
        
    else:
        images = "" # glob search for images in the folder
        # for loop through images
        # predict image with model from config file
        prediction = ""
        # store files 
    return 
if __name__ == "__main__":  
    parser = argparse.ArgumentParser("Inference to predict images with trained STACC model.")
    parser.add_argument("config", help="Please enter STACC config file.")
    parser.add_argument("prediction_path", help="Please provide a path to a folder with JPG or TIF images or a path to an image.")
    parser.add_argument("-s", "store_path", required=False, help="(Optional) Provide path where prediction should be stored.")
    args = parser.parse_args()

    config_path = args.config
    prediction_path = args.prediction_path
    store_path = args.store_path

    main(config_path, prediction_path, store_path)
