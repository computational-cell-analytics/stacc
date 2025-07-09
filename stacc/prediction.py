import os
from glob import glob

import imageio.v3 as imageio
import numpy as np
import torch
from skimage.feature import peak_local_max

from .util import get_model, standardize, get_postprocessing_parameters


def _pad_image(input_, min_divisible_):
    if any(sh % md != 0 for sh, md in zip(input_.shape, min_divisible_)):
        pad_width = tuple((0, 0 if sh % md == 0 else md - sh % md) for sh, md in zip(input_.shape, min_divisible_))
        crop_padding = tuple(slice(0, sh) for sh in input_.shape)
        input_ = np.pad(input_, pad_width, mode="reflect")
    else:
        crop_padding = None
    return input_, crop_padding


def run_counting(
    model: torch.nn.Module,
    image: np.ndarray,
    min_distance: float = 10,
    threshold_abs: float = 1.0,
) -> np.ndarray:
    """Predict object centroid coordinates for an input image.

    Args:
        model: The U-Net trained for the counting task.
        image: The image data.
        min_distance: The minimal distance between detected objects, in pixels.
        threshold_abs: The threshold for detecting an object, with respect to the output predictions.

    Returns:
        The coordinates of predicted objects.
    """
    model.eval()

    # Process the image for our model so that it has the dimensions C x Y x X.
    if image.ndim == 3 and image.shape[-1] == 3:  # RGB image
        image = image.transpose((2, 0, 1))
    elif image.ndim == 2:
        image = image[None]
    assert image.ndim == 3

    # Check if the number of channels in the image match the model inputs.
    n_channels = image.shape[0]
    if n_channels != model.in_channels:
        raise RuntimeError(
            f"Wrong number of channels: The number of channels in your image is {n_channels}, "
            f"which does not match the number of input channels of the model, which is {model.in_channels}."
        )

    # Normalize the image.
    image = standardize(image)

    # Pad the image if necessary.
    min_divisible = [1, 16, 16]
    image, crop_padding = _pad_image(image, min_divisible)

    with torch.no_grad():
        input_ = torch.from_numpy(image[None])
        prediction = model(input_)
        prediction = prediction.numpy().squeeze()
    assert prediction.ndim == 2

    if crop_padding is not None:
        prediction = prediction[crop_padding[1:]]

    predicted_coords = peak_local_max(prediction, min_distance=min_distance, threshold_abs=threshold_abs)
    return predicted_coords


def run_counting_stacked(
    model: torch.nn.Module,
    image_stack: np.ndarray,
    min_distance: float = 10,
    threshold_abs: float = 1.0,
) -> np.ndarray:
    """Predict object centroid coordinates for an image stack.

    Args:
        model: The U-Net trained for the counting task.
        image_stack: The image data with 3 dimensions.
        min_distance: The minimal distance between detected objects, in pixels.
        threshold_abs: The threshold for detecting an object, with respect to the output predictions.

    Returns:
        The coordinates of predicted objects.
    """
    predicted_coords = []
    for i, frame in enumerate(image_stack):
        frame_coords = run_counting(model, frame, min_distance, threshold_abs)
        if len(frame_coords) == 0:
            continue
        frame_coords = np.concatenate([np.full(frame_coords.shape[0], i)[:, None], frame_coords], axis=1)
        predicted_coords.append(frame_coords)
    if len(predicted_coords) == 0:
        return np.zeros((0, 3), dtype="int")
    predicted_coords = np.concatenate(predicted_coords)
    return predicted_coords


def _get_inputs(input_, pattern):
    if not os.path.exists(input_):
        raise ValueError(f"Invalid input path {input_}")

    if os.path.isfile(input_):
        return [input_]

    inputs = glob(os.path.join(input_, pattern))
    return inputs


def main():
    """@private"""
    import argparse
    import pandas as pd
    from pathlib import Path
    import imageio
    import os

    parser = argparse.ArgumentParser(
        description="Count the number of colonies or cells in an image or in multiple images. "
        "For example, you can run 'stacc.counting -i images/colony_example_image.jpg' to count "
        "the colonies in the colony example image. Or 'stacc.counting -i images/cell_example_image.png' "
        "to count the cells in the cell example image."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="The filepath to the input. This can either point to an image or a folder with images.",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="*",
        help="A pattern to select images from the input folder. For example, you can pass '*.png' to select only png images.",  # noqa
    )
    parser.add_argument(
        "-m",
        "--model",
        default="colonies",
        help="The model to use for counting. Can either be 'colonies' for colony counting or 'cells' for cell counting."
        " By default the model for colony counting is used.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="The output folder for saving results. If it is given a csv with the centroids of counted objects will be saved for each image.",
    )
    parser.add_argument("--custom_model", help="Path to a custom trained model.")
    parser.add_argument(
        "--custom_distance", type=int, help="Corresponding min_distance value for postprocessing the custom model."
    )
    parser.add_argument(
        "--custom_threshold", type=float, help="Corresponding threshold_abs value for postprocessing the custom model."
    )
    args = parser.parse_args()

    inputs = _get_inputs(args.input, args.pattern)
    print("Counting", args.model, "in", len(inputs), "image(s).")

    model = get_model(args.model, args.custom_model)

    # check that all arguments are given for custom model
    if args.custom_model and args.custom_distance is not None and args.custom_threshold is not None:
        min_distance = args.custom_distance
        threshold_abs = args.custom_threshold
    else:
        min_distance, threshold_abs = get_postprocessing_parameters(args.model)

    for image_path in inputs:
        image = imageio.imread(image_path)
        prediction = run_counting(model, image, min_distance=min_distance, threshold_abs=threshold_abs)

        count = len(prediction)
        print("The count for", image_path, "is:", count)

        if args.output is not None:
            os.makedirs(args.output, exist_ok=True)
            fname = Path(image_path).stem
            out_path = os.path.join(args.output, f"{fname}.csv")

            df = pd.DataFrame(prediction, columns=["y", "x"])
            df.to_csv(out_path, index=False)
