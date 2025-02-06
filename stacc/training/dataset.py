import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from skimage.io import imread
from skimage.filters import gaussian
from torch_em.util import ensure_spatial_array, ensure_tensor_with_channels, load_image
from torch.utils.data import DataLoader


def _get_image_id(image_path: str) -> str:
    """Extract the image ID from the given image path by removing the file extension.

    This function checks if the image is in JPG or TIF format. If not, it raises a ValueError.

    Args:
        image_path: The file path of the image.

    Returns:
        The image ID, which is the basename of the image path without its extension.

    Raises:
        ValueError: If the image is not a JPG or TIF file.
    """
    # Get the file extension
    _, file_extension = os.path.splitext(image_path)

    # Check if the file is a JPG or TIF image
    if file_extension.lower() not in [".jpg", ".jpeg", ".tif", ".tiff"]:
        raise ValueError("The image must be a JPG or TIF file.")

    # Get the basename without the extension
    return os.path.splitext(os.path.basename(image_path))[0]


def apply_stamp_to_stacc_gt_label(
    stamp: np.ndarray,
    position: Tuple[int, int],
    label_matrix: np.ndarray,
    image_id: str
) -> None:
    """Add a stamp (small matrix) at the coordinates 'position' of the label_matrix.

    The stamp is added to the label_matrix in-place, using the maximum value if the label_matrix
    already has a non-zero value at that position.

    Args:
        stamp: The stamp matrix to be applied.
        position: The (x, y) coordinates where the center of the stamp should be placed.
        label_matrix: The stacc ground truth label matrix to which the stamp is applied.
        image_id: The image identifier for error reporting.
    """
    w_stamp, _ = stamp.shape  # stamp has square shape, w_stamp is odd
    x, y = position
    h = w_stamp // 2
    lx, ly = label_matrix.shape

    # Calculate the bounds for the stamp application
    x_start = max(x - h, 0)
    x_end = min(x + h + 1, lx)
    y_start = max(y - h, 0)
    y_end = min(y + h + 1, ly)

    # Calculate the bounds for the stamp itself
    stamp_x_start = max(h - x, 0)
    stamp_x_end = w_stamp - max(x + h + 1 - lx, 0)
    stamp_y_start = max(h - y, 0)
    stamp_y_end = w_stamp - max(y + h + 1 - ly, 0)

    try:
        # Apply the stamp to the label matrix
        a = label_matrix[x_start:x_end, y_start:y_end]
        label_matrix[x_start:x_end, y_start:y_end] = np.maximum(
            a, stamp[stamp_x_start:stamp_x_end, stamp_y_start:stamp_y_end]
        )
    except Exception as e:
        print(f"An error occurred in image {image_id}.")
        print(f"Warning: Corresponding error message: {e}.")


def width_to_sigma(width: int, lower_bound: float, upper_bound: float, eps: float = 0.00001) -> float:
    """Convert a given width to a Gaussian sigma value used for the stamp, ensuring it is within specified bounds.

    Args:
        width: The width of the Gaussian stamp, this is what we translate to a sigma value.
        lower_bound: The minimum allowable value for sigma.
        upper_bound: The maximum allowable value for sigma.
        eps: A small epsilon value used in the logarithmic calculation for truncating.

    Returns:
        The calculated sigma value, bounded by lower_bound and upper_bound.
    """
    sigma = np.sqrt(-(width**2) / (2 * np.log(eps)))

    # Ensure sigma is within the specified bounds
    if lower_bound and upper_bound:
        if sigma < lower_bound:
            sigma = lower_bound
        elif sigma > upper_bound:
            sigma = upper_bound

    return sigma


def create_gaussian_stamp(width: int, lower_bound: float, upper_bound: float, eps: float = 0.00001) -> np.ndarray:
    """Create a round Gaussian stamp (matrix) with size width x width.

    Args:
        width: The width of the Gaussian stamp.
        lower_bound: The minimum allowable value for sigma.
        upper_bound: The maximum allowable value for sigma.
        eps: A small epsilon value used for Gaussian truncation. Default is 0.00001.

    Returns:
        A 2D array representing the Gaussian stamp.
    """
    sigma = width_to_sigma(width, lower_bound, upper_bound, eps)

    stamp = np.zeros((width, width))
    stamp[width // 2, width // 2] = 1
    stamp = gaussian(stamp, sigma=sigma, truncate=10.0, mode='constant')
    stamp[np.where(stamp < eps)] = 0  # Truncate to make the stamp circular
    stamp = stamp * 2 * 4 * np.pi * sigma**2

    return stamp


def create_stacc_ground_truth_label_from_json(
    image_path: str,
    label_path: str,
    eps: float = 0.00001,
    sigma: Optional[float] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
):
    """Create a ground truth label matrix from JSON bounding box annotations.

    Args:
        image_path: Path to the input image.
        label_path: Path to the corresponding JSON file containing bounding box labels.
        eps: Epsilon value for Gaussian truncation. Default is 0.00001.
        sigma: Sigma value for Gaussian blur. If None, individual stamps are applied.
        lower_bound: Lower bound for Gaussian sigma. If None, no lower bound for sigma will be set.
        upper_bound: Upper bound for Gaussian sigma If None, no upper bound for sigma will be set.

    Returns:
        A 2D array representing the stacc ground truth label matrix.
    """
    with open(label_path) as file:
        label_dict = json.load(file)

    bboxes = label_dict["labels"]
    n_colonies = len(bboxes)  # Number of annotations / bounding boxes

    image_id = _get_image_id(image_path)  # Get image id and check if jpg or tif image
    im = imread(image_path)

    n_rows, n_columns = im.shape[:2]

    stacc_gt_label = np.zeros((n_rows, n_columns))  # Create empty ground truth label
    if n_colonies > 0:
        # Only keep the stacc_gt_label that are inside the image dimensions
        reasonable_indices = [
            i for i in range(n_colonies) if
            (bboxes[i]["x"] + max(int(bboxes[i]["width"]/2), 1)) <= n_columns and
            (bboxes[i]["y"] + max(int(bboxes[i]["height"]/2), 1)) <= n_rows
        ]
        x_coordinates = np.array(
            [(int(bboxes[i]["y"]) + max(int(bboxes[i]["width"]/2), 1)) for i in reasonable_indices], dtype="int"
        )
        y_coordinates = np.array(
            [(int(bboxes[i]["x"]) + max(int(bboxes[i]["height"]/2), 1)) for i in reasonable_indices], dtype="int"
        )

        if sigma:
            # Process all coordinates at once
            stacc_gt_label[x_coordinates, y_coordinates] = 1
            stacc_gt_label = gaussian(stacc_gt_label, sigma=sigma, mode="constant")
            stacc_gt_label[np.where(stacc_gt_label < eps)] = 0
            stacc_gt_label = stacc_gt_label * 2 * 4 * np.pi * sigma**2
        else:
            # Process each coordinate individually
            for i, (x_coord, y_coord) in enumerate(zip(x_coordinates, y_coordinates)):
                width = max(int(bboxes[reasonable_indices[i]]['width']), 1)
                height = max(int(bboxes[reasonable_indices[i]]['height']), 1)
                width = min(width, height)  # Make the stamp square
                if width % 2 == 0:
                    width -= 1
                stamp = create_gaussian_stamp(width, lower_bound, upper_bound, eps)
                coords = (x_coord, y_coord)
                apply_stamp_to_stacc_gt_label(stamp, position=coords, label_matrix=stacc_gt_label, image_id=image_id)

        return stacc_gt_label
    else:
        return stacc_gt_label


def create_stacc_labels_from_csv(
    image_path: str,
    csv_path: str,
    sigma: float,
    eps: float = 0.00001,
) -> np.ndarray:
    """Create ground truth labels from CSV point annotations.

    Args:
        image_path: Path to the input image.
        csv_path: Path to the CSV file containing point coordinates.
        sigma: Sigma value for Gaussian blur.
        eps: Epsilon value for Gaussian truncation. Default is 0.00001.

    Returns:
        A 2D array representing the stacc ground truth label matrix.
    """
    df = pd.read_csv(csv_path)
    x_coordinates = np.round(df["axis-0"].values).astype(int)
    y_coordinates = np.round(df["axis-1"].values).astype(int)

    im = imread(image_path)
    n_rows, n_columns = im.shape[:2]

    stacc_gt_label = np.zeros((n_rows, n_columns))
    if len(x_coordinates) > 0:
        stacc_gt_label[x_coordinates, y_coordinates] = 1
        stacc_gt_label = gaussian(stacc_gt_label, sigma=sigma, mode="constant")
        stacc_gt_label[np.where(stacc_gt_label < eps)] = 0
        stacc_gt_label = stacc_gt_label * 2 * 4 * np.pi * sigma**2

    return stacc_gt_label


class StaccImageCollectionDataset(torch.utils.data.Dataset):
    """@private
    """
    max_sampling_attempts = 500

    def _check_inputs(self, raw_images, label_images):
        if len(raw_images) != len(label_images):
            raise ValueError(
                f"Expect same number of raw and label images, got {len(raw_images)} and {len(label_images)}"
            )

    def __init__(
        self,
        raw_image_paths,
        label_image_paths,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        label_transform2=None,
        transform=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None,
        eps=1e-5,
        sigma=None,
        lower_bound=None,
        upper_bound=None,
    ):
        self._check_inputs(raw_image_paths, label_image_paths)
        self.raw_images = raw_image_paths
        self.label_images = label_image_paths
        self._ndim = 2

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

        # Julias arguments!
        self.eps = eps
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if n_samples is None:
            self._len = len(self.raw_images)
            self.sample_random_index = False
        else:
            self._len = n_samples
            self.sample_random_index = True

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, shape):
        if any(sh < psh for sh, psh in zip(shape, self.patch_shape)):
            raise NotImplementedError(
                f"Image padding is not supported yet. Data shape {shape}, patch shape {self.patch_shape}"
            )
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _get_sample(self, index):
        if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))
        # these are just the file paths
        raw_path, label_path = self.raw_images[index], self.label_images[index]

        # print(f"Raw path: {raw}, label path: {label}")

        raw = load_image(raw_path)

        _, label_extension = os.path.splitext(label_path)
        if label_extension == ".json":
            label = create_stacc_ground_truth_label_from_json(
                raw_path, label_path, eps=self.eps, sigma=self.sigma,
                lower_bound=self.lower_bound, upper_bound=self.upper_bound
            )
        elif label_extension.lower() == ".csv":
            if self.sigma is None:
                raise RuntimeError("Training from CSV labels requires a sigma value.")
            label = create_stacc_labels_from_csv(raw_path, label_path, sigma=self.sigma, eps=self.eps)
        else:
            raise ValueError(f"Unsupported label file extension: {label_extension}")

        # print(f"type of raw: {type(raw)}, type of label: {type(label)}")

        have_raw_channels = raw.ndim == 3
        have_label_channels = label.ndim == 3
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")

        shape = raw.shape
        # we determine if image has channels as te first or last axis base on array shape.
        # This will work only for images with less than 16 channels.
        prefix_box = tuple()
        if have_raw_channels:
            # use heuristic to decide whether the data is stored in channel last or channel first order:
            # if the last axis has a length smaller than 16 we assume that it's the channel axis,
            # otherwise we assume it's a spatial axis and that the first axis is the channel axis.
            if shape[-1] < 16:
                shape = shape[:-1]
            else:
                shape = shape[1:]
                prefix_box = (slice(None), )

        # sample random bounding box for this image
        bb = self._sample_bounding_box(shape)
        # print(f"bb: {bb}")
        raw_patch = np.array(raw[prefix_box + bb])
        label_patch = np.array(label[bb])

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw_patch, label_patch):
                bb = self._sample_bounding_box(shape)
                raw_patch = np.array(raw[prefix_box + bb])
                label_patch = np.array(label[bb])
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        # to channel first
        if have_raw_channels and len(prefix_box) == 0:
            raw_patch = raw_patch.transpose((2, 0, 1))

        return raw_patch, label_patch

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)
            # if self.trafo_halo is not None:
            #     raw = self.crop(raw)
            #     labels = self.crop(labels)

        # support enlarging bounding box here as well (for affinity transform) ?
        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels


def _split_data_paths_into_training_dataset(dataset_file):
    if isinstance(dataset_file, str):
        with open(dataset_file) as dataset:
            dict_dataset = json.load(dataset)
    else:
        assert isinstance(dataset_file, dict)
        dict_dataset = dataset_file

    train_images = dict_dataset["train"]["images"]
    train_labels = dict_dataset["train"]["labels"]

    val_images = dict_dataset["val"]["images"]
    val_labels = dict_dataset["val"]["labels"]

    if "test" in dict_dataset:
        test_images = dict_dataset["test"]["images"]
        test_labels = dict_dataset["test"]["labels"]
    else:
        test_images, test_labels = None, None

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def get_stacc_data_loader(
    train_dataset_file: Union[str, Dict[str, Dict[str, List[str]]]],
    patch_shape: Tuple[int, ...],
    n_workers: int,
    batch_size: int,
    eps: float = 1e-5,
    sigma: Optional[float] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get training, validation and test data loader for a STACC model.

    Args:
        train_dataset_file: Path to a JSON file with file paths for the train, val and test splits.
            Can also be a dictionary with the respective file paths. The test split is optional.
        patch_shape: The patch shape for the data loaders.
        n_workers: Number of workers for the data loaders.
        batch_size: The batch size for training.
        eps: A small epsilon value used in the logarithmic calculation for truncating.
        sigma: Sigma value for Gaussian blur. If None, individual stamps are applied.
        lower_bound: The minimum allowable value for sigma.
        upper_bound: The maximum allowable value for sigma.

    Returns:
        The data loader for the training split.
        The data loader for the validation split.
        The data loader for the test split.
    """

    train_images, train_labels, val_images, val_labels, test_images, test_labels =\
        _split_data_paths_into_training_dataset(train_dataset_file)

    train_set = StaccImageCollectionDataset(train_images, train_labels, patch_shape, eps=eps, sigma=sigma,
                                            lower_bound=lower_bound, upper_bound=upper_bound)
    val_set = StaccImageCollectionDataset(val_images, val_labels, patch_shape, eps=eps, sigma=sigma,
                                          lower_bound=lower_bound, upper_bound=upper_bound)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    train_dataloader.shuffle = True
    val_dataloader.shuffle = True

    if test_images is None:
        test_dataloader = None
    else:
        test_set = StaccImageCollectionDataset(test_images, test_labels, patch_shape, eps=eps, sigma=sigma,
                                               lower_bound=lower_bound, upper_bound=upper_bound)
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        test_dataloader.shuffle = True

    return train_dataloader, val_dataloader, test_dataloader
