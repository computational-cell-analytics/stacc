import os
from glob import glob

from multiprocessing import cpu_count
from typing import Optional, Tuple, Union

import imageio.v3 as imageio
import numpy as np

from sklearn.model_selection import train_test_split

from .dataset import get_stacc_data_loader, width_to_sigma
from .stacc_training import run_stacc_training
from .util import export_model
from ..util import get_model_path


def _get_training_dict(
    image_folder, label_folder, image_pattern, label_pattern, validation_fraction
):
    images = sorted(glob(os.path.join(image_folder, image_pattern)))
    labels = sorted(glob(os.path.join(label_folder, label_pattern)))
    if len(images) != len(labels):
        raise ValueError(f"Number of image and label paths do not match: {len(images)} != {len(labels)}.")

    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=validation_fraction, random_state=42,
    )

    return {
        "train": {"images": train_images, "labels": train_labels},
        "val": {"images": val_images, "labels": val_labels}
    }


def _derive_patch_shape(training_dict, min_divisible_shape=16):
    images = training_dict["train"]["images"] + training_dict["val"]["images"]
    shapes = []
    for image_path in images:
        shape = imageio.imread(image_path).shape
        shapes.append(shape)
    shapes = np.array(shapes)
    patch_shape = tuple(np.min(shapes, axis=0).tolist())

    # Get the max shape that fits patch shape and that is divisible by 16.
    patch_shape = tuple(ps - ps % min_divisible_shape for ps in patch_shape)
    return patch_shape


def run_stacc_training_from_napari_annotations(
    name: str,
    pretrained_model_name: Optional[str],
    image_folder: str,
    label_folder: str,
    average_object_width: float,
    image_pattern: str = "*",
    label_pattern: str = "*.csv",
    patch_shape: Optional[Tuple[int, int]] = None,
    validation_fraction: float = 0.1,
    n_workers: Optional[int] = None,
    batch_size: int = 1,
    n_epochs: int = 25,
    save_new_model_path: Optional[Union[str, os.PathLike]] = None,
):
    """Train a counting model from images and point annotations extracted from napari.

    Args:
        name: The name of the model to be trained.
        pretrained_model_name: The name of the model to use for weight initialization.
            Can be of of 'cells', 'colonies', or None (for training from scratch).
        image_folder: The folder with the images for training.
        label_folder: The folder with the csv files for training.
        average_object_width: The average width of an object in the training data (in pixel).
        image_pattern: A wildcard pattern for loading images from `image_folder`.
            By default all files in the folder will be loaded.
        label_pattern: A wildcard pattern for loading csv files from `label_folder`.
            By default all csv files in the folder will be loaded.
        patch_shape: The patch shape for training.
            By default the smallest shape that fits all images will be used.
        validation_fraction: The fraction of images to use for the validation set.
        n_workers: The number of workers for the dataloaders.
        batch_size: The batch size for training.
        n_epochs: The number of epochs to use for training.
        save_new_model_path: Path to save the new model.
    """
    training_dict = _get_training_dict(
        image_folder, label_folder, image_pattern, label_pattern, validation_fraction
    )

    # If we don't have a patch shape then we use the full image shape.
    # We find the smallest image shape and use it as a patch shape,
    # rounded down to a common denominator of 16.
    if patch_shape is None:
        patch_shape = _derive_patch_shape(training_dict)

    if n_workers is None:
        n_workers = cpu_count()

    sigma = width_to_sigma(average_object_width, lower_bound=None, upper_bound=None)
    train_loader, val_loader, _ = get_stacc_data_loader(
        training_dict, n_workers=n_workers, patch_shape=patch_shape, batch_size=batch_size, sigma=sigma,
    )

    # If the pretrained model name is None, then we train from scratch.
    # Otherwise we get the path to the respective model weights
    # (currently either cells or colonies).
    if pretrained_model_name is None:
        pretrained_model_path = None
    else:
        pretrained_model_path = get_model_path(pretrained_model_name)

    run_stacc_training(
        model_name=name, train_loader=train_loader, val_loader=val_loader,
        n_epochs=n_epochs, pretrained_model_path=pretrained_model_path,
        save_new_model_path=save_new_model_path,
    )

    # Export the model, so that it can be used directly in napari or for batched prediction.
    export_root = "" if save_new_model_path is None else save_new_model_path
    checkpoint_path = os.path.join(export_root, "checkpoints", name, "best.pt")
    export_path = os.path.join(export_root, f"{name}.pt")
    print("The trained model was exported to", export_path)
    export_model(checkpoint_path, export_path)


def main():
    """@private
    """
    import argparse

    parser = argparse.ArgumentParser("Train a counting model from images and point annotations extracted from napari.")
    parser.add_argument("--name", "-n", help="The name of the model to be trained.", required=True)
    parser.add_argument(
        "--pretrained_model_name", "-p",
        help="The name of the model to use for weight initialization. Can be of of 'cells', 'colonies'. "
        "If not given, the model will be trained from scratch."
    )
    parser.add_argument("--image_folder", "-i", required=True, help="The folder with the images for training.")
    parser.add_argument("--label_folder", "-l", required=True, help="The folder with the csv files for training.")
    parser.add_argument(
        "--average_object_width", "-o", required=True, type=float,
        help="The average width of an object in the training data (in pixel)."
    )
    parser.add_argument(
        "--image_pattern", default="*",
        help="A wildcard pattern for loading images from `image_folder`. "
        "By default all files in the folder will be loaded."
    )
    parser.add_argument(
        "--label_pattern", default="*.csv",
        help="A wildcard pattern for loading csv files from `label_folder`. "
        "By default all csv files in the folder will be loaded."
    )
    parser.add_argument(
        "--patch_shape", type=int, nargs=2,
        help="The patch shape for training. By default the smallest shape that fits all images will be used."
    )
    parser.add_argument(
        "--validation_fraction", type=float, help="The fraction of images to use for the validation set."
    )
    parser.add_argument("--n_workers", type=int, help="The number of workers for the dataloaders.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training.")
    parser.add_argument("--n_epochs", type=int, default=25, help="The number of epochs to use for training.")
    parser.add_argument("--save_new_model_path", help="Path to save the trained model.")
    args = parser.parse_args()

    run_stacc_training_from_napari_annotations(
        name=args.name, pretrained_model_name=args.pretrained_model_name,
        image_folder=args.image_folder, label_folder=args.label_folder,
        average_object_width=args.average_object_width,
        image_pattern=args.image_pattern, label_pattern=args.label_pattern,
        patch_shape=args.patch_shape, validation_fraction=args.validation_fraction,
        n_workers=args.n_workers, batch_size=args.batch_size,
        n_epochs=args.n_epochs, save_new_model_path=args.save_new_model_path,
    )
