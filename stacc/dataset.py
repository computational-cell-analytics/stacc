import torch
import json
import os
import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.filters import gaussian
from torch_em.util import (ensure_spatial_array, ensure_tensor_with_channels, load_image, supports_memmap)
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# for testing
import matplotlib.pyplot as plt


def _get_image_id(image_path):
    """
    Extracts the image ID from the given image path by removing the file extension.
    This function checks if the image is in JPG or TIF format. If not, it raises a ValueError.

    Args:
        image_path (str): The file path of the image.

    Returns:
        str: The image ID, which is the basename of the image path without its extension.

    Raises:
        ValueError: If the image is not a JPG or TIF file.
    """
    _, file_extension = os.path.splitext(image_path)

    if file_extension.lower() not in ['.jpg', '.jpeg', '.tif', '.tiff']:
        raise ValueError("The image must be a JPG or TIF file.")

    image_id = os.path.splitext(os.path.basename(image_path))[0]

    return image_id


def _split_data_paths_into_training_dataset(dataset_file):
    """
    Splits data paths from a dataset file into training, validation, and test sets.

    Args:
        dataset_file (str): Path to the JSON file containing dataset paths.

    Returns:
        tuple: Lists of file paths for train images, train labels, validation images, validation labels, 
               test images, and test labels.
    """
    with open(dataset_file) as dataset:
        dict_dataset = json.load(dataset)

    train_images = dict_dataset['train']['images']
    train_labels = dict_dataset['train']['labels']

    val_images = dict_dataset['val']['images']
    val_labels = dict_dataset['val']['labels']

    test_images = dict_dataset['test']['images']
    test_labels = dict_dataset['test']['labels']

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def _split_csv_data(images, annotations, test_size=0.15, val_size=0.05, random_state=42):
    """
    Splits image and annotation data into training, validation, and test sets.

    Args:
        images (list): List of image file paths.
        annotations (list): List of annotation file paths.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.15.
        val_size (float, optional): Proportion of the dataset to include in the validation split. Default is 0.05.
        random_state (int, optional): Random seed for shuffling the data. Default is 42.

    Returns:
        tuple: Lists of file paths for train images, train labels, validation images, validation labels, 
               test images, and test labels.
    """
    images.sort()
    annotations.sort()

    assert len(images) == len(annotations), "Mismatch in number of images and annotations"

    image_filenames = [os.path.splitext(os.path.basename(f))[0] for f in images]
    annotation_filenames = [os.path.splitext(os.path.basename(f))[0] for f in annotations]
    assert image_filenames == annotation_filenames, "Mismatch in filenames between images and annotations"

    matched_pairs = list(zip(images, annotations))
    train_val_pairs, test_pairs = train_test_split(matched_pairs, test_size=test_size, random_state=random_state)
    val_size_relative = val_size / (1 - test_size)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_size_relative, random_state=random_state)

    train_images, train_labels = zip(*train_pairs)
    val_images, val_labels = zip(*val_pairs)
    test_images, test_labels = zip(*test_pairs)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def apply_stamp_to_stacc_gt_label(stamp, position, label_matrix, image_id):
    """
    Adds a stamp (small matrix) at the coordinates 'position' of the label_matrix.
    The stamp is added to the label_matrix, using the maximum value if the label_matrix
    already has a non-zero value at that position.

    Args:
        stamp (np.ndarray): The stamp matrix to be applied.
        position (tuple): The (x, y) coordinates where the center of the stamp should be placed.
        label_matrix (np.ndarray): The stacc ground truth label matrix to which the stamp is applied.
        image_id (str): The image identifier for error reporting.
    """
    w_stamp, _ = stamp.shape
    x, y = position
    h = w_stamp // 2
    lx, ly = label_matrix.shape

    x_start = max(x - h, 0)
    x_end = min(x + h + 1, lx)
    y_start = max(y - h, 0)
    y_end = min(y + h + 1, ly)

    stamp_x_start = max(h - x, 0)
    stamp_x_end = w_stamp - max(x + h + 1 - lx, 0)
    stamp_y_start = max(h - y, 0)
    stamp_y_end = w_stamp - max(y + h + 1 - ly, 0)

    try:
        a = label_matrix[x_start:x_end, y_start:y_end]
        label_matrix[x_start:x_end, y_start:y_end] = np.maximum(
            a, stamp[stamp_x_start:stamp_x_end, stamp_y_start:stamp_y_end]
        )
    except Exception as e:
        print(f"An error occurred in image {image_id}.")
        print(f"Warning: Corresponding error message: {e}.")

    return


def width_to_sigma(width, lower_bound, upper_bound, eps=0.00001):
    """
    Converts a given width to a Gaussian sigma value used for the stamp, ensuring it is within specified bounds.

    Args:
        width (int): The width of the Gaussian stamp, this is what we translate to a sigma value.
        lower_bound (float): The minimum allowable value for sigma.
        upper_bound (float): The maximum allowable value for sigma.
        eps (float, optional): A small epsilon value used in the logarithmic calculation for trucating. Default is 0.00001.

    Returns:
        float: The calculated sigma value, bounded by lower_bound and upper_bound.
    """
    sigma = np.sqrt(-(width**2) / (2 * np.log(eps)))

    if lower_bound and upper_bound:
        if sigma < lower_bound:
            sigma = lower_bound
        elif sigma > upper_bound:
            sigma = upper_bound

    return sigma


def create_gaussian_stamp(width, lower_bound, upper_bound, eps=0.00001):
    """
    Creates a round Gaussian stamp (matrix) with size width x width.

    Args:
        width (int): The width of the Gaussian stamp.
        lower_bound (float): The minimum allowable value for sigma.
        upper_bound (float): The maximum allowable value for sigma.
        eps (float, optional): A small epsilon value used for Gaussian truncation. Default is 0.00001.

    Returns:
        np.ndarray: A 2D array representing the Gaussian stamp.
    """
    sigma = width_to_sigma(width, lower_bound, upper_bound, eps)

    stamp = np.zeros((width, width))
    stamp[width // 2, width // 2] = 1
    stamp = gaussian(stamp, sigma=sigma, truncate=10.0, mode='constant')
    stamp[np.where(stamp < eps)] = 0
    stamp = stamp * 2 * 4 * np.pi * sigma**2

    return stamp


def create_stacc_ground_truth_label_from_json(image_path, label_path, eps=0.00001, sigma=None, lower_bound=None, upper_bound=None):
    """
    Creates a ground truth label matrix from JSON bounding box annotations.

    Args:
        image_path (str): Path to the input image.
        label_path (str): Path to the corresponding JSON file containing bounding box labels.
        eps (float, optional): Epsilon value for Gaussian truncation. Default is 0.00001.
        sigma (float, optional): Sigma value for Gaussian blur. If None, individual stamps are applied.
        lower_bound (float, optional): Lower bound for Gaussian sigma. If None, no lower bound for sigma will be set.
        upper_bound (float, optional): Upper bound for Gaussian sigma If None, no upper bound for sigma will be set.

    Returns:
        np.ndarray: A 2D array representing the stacc ground truth label matrix.
    """
    with open(label_path) as file:
        label_dict = json.load(file)

    bboxes = label_dict['labels']
    n_colonies = len(bboxes)

    image_id = _get_image_id(image_path)
    im = imread(image_path)

    n_rows, n_columns = im.shape[:2]

    stacc_gt_label = np.zeros((n_rows, n_columns))
    if n_colonies > 0:
        reasonable_indices = [
            i for i in range(n_colonies)
            if (bboxes[i]['x'] + max(int(bboxes[i]['width'] / 2), 1)) <= n_columns and
               (bboxes[i]['y'] + max(int(bboxes[i]['height'] / 2), 1)) <= n_rows
        ]
        x_coordinates = np.array(
            [(int(bboxes[i]['y']) + max(int(bboxes[i]['width'] / 2), 1)) for i in reasonable_indices], dtype='int'
        )
        y_coordinates = np.array(
            [(int(bboxes[i]['x']) + max(int(bboxes[i]['height'] / 2), 1)) for i in reasonable_indices], dtype='int'
        )

        if sigma:
            stacc_gt_label[x_coordinates, y_coordinates] = 1
            stacc_gt_label = gaussian(stacc_gt_label, sigma=sigma, mode="constant")
            stacc_gt_label[np.where(stacc_gt_label < eps)] = 0
            stacc_gt_label = stacc_gt_label * 2 * 4 * np.pi * sigma**2
        else:
            for i, (x_coord, y_coord) in enumerate(zip(x_coordinates, y_coordinates)):
                width = max(int(bboxes[reasonable_indices[i]]['width']), 1)
                height = max(int(bboxes[reasonable_indices[i]]['height']), 1)
                width = min(width, height)
                if width % 2 == 0:
                    width -= 1
                stamp = create_gaussian_stamp(width, lower_bound, upper_bound, eps)
                coords = (x_coord, y_coord)
                apply_stamp_to_stacc_gt_label(stamp, position=coords, label_matrix=stacc_gt_label, image_id=image_id)

        return stacc_gt_label
    else:
        return stacc_gt_label


def create_stacc_labels_from_csv(image_path, csv_path, average_object_width=20, eps=0.00001):
    """
    Creates a ground truth label matrix from CSV point annotations.

    Args:
        image_path (str): Path to the input image.
        csv_path (str): Path to the CSV file containing point coordinates.
        average_object_width (float): Average width of the objects to determine the Gaussian sigma.
        eps (float, optional): Epsilon value for Gaussian truncation. Default is 0.00001.

    Returns:
        np.ndarray: A 2D array representing the stacc ground truth label matrix.
    """
    df = pd.read_csv(csv_path)
    x_coordinates = df['axis-0'].values
    y_coordinates = df['axis-1'].values

    im = imread(image_path)

    n_rows, n_columns = im.shape[:2]

    stacc_gt_label = np.zeros((n_rows, n_columns))

    if len(x_coordinates) > 0:
        sigma = np.sqrt(-(average_object_width**2) / (2 * np.log(eps)))

        stacc_gt_label[x_coordinates, y_coordinates] = 1
        stacc_gt_label = gaussian(stacc_gt_label, sigma=sigma, mode="constant")
        stacc_gt_label[np.where(stacc_gt_label < eps)] = 0
        stacc_gt_label = stacc_gt_label * 2 * 4 * np.pi * sigma**2

    return stacc_gt_label


import torch
import numpy as np
import os

class StaccImageCollectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    def _check_inputs(self, raw_images, label_images):
        if len(raw_images) != len(label_images):
            raise ValueError(
                f"Expect same number of raw and label images, got {len(raw_images)} and {len(label_images)}"
            )
        first_image_shape = None
        for raw_im, label_im in zip(raw_images, label_images):
            if supports_memmap(raw_im) and supports_memmap(label_im):
                shape = load_image(raw_im).shape
                if first_image_shape is None:
                    first_image_shape = shape
                else:
                    if shape != first_image_shape:
                        raise ValueError("All images must have the same dimensions.")
                if any(dim % 16 != 0 for dim in shape):
                    raise ValueError("Image dimensions must be divisible by 16.")

                assert len(shape) in (2, 3)

    def __init__(
        self,
        raw_image_paths,
        label_image_paths,
        patch_shape=None,
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
        average_object_width=20
    ):
        self._check_inputs(raw_image_paths, label_image_paths)
        self.raw_images = raw_image_paths
        self.label_images = label_image_paths
        self._ndim = 2

        # Assign patch_shape to self.patch_shape
        if patch_shape is None:
            first_image_shape = load_image(raw_image_paths[0]).shape
            self.patch_shape = first_image_shape[:2]  # Use only the spatial dimensions
        else:
            self.patch_shape = patch_shape

        assert len(self.patch_shape) == self._ndim

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

        self.eps = eps
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.average_object_width = average_object_width

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
        raw_path, label_path = self.raw_images[index], self.label_images[index]

        raw = load_image(raw_path)

        _, label_extension = os.path.splitext(label_path)
        if label_extension.lower() == '.json':
            label = create_stacc_ground_truth_label_from_json(
                raw_path, label_path, eps=self.eps, sigma=self.sigma,
                lower_bound=self.lower_bound, upper_bound=self.upper_bound
            )
        elif label_extension.lower() == '.csv':
            label = create_stacc_labels_from_csv(
                raw_path, label_path, average_object_width=self.average_object_width, eps=self.eps
            )
        else:
            raise ValueError(f"Unsupported label file extension: {label_extension}")

        have_raw_channels = raw.ndim == 3
        have_label_channels = label.ndim == 3
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")

        shape = raw.shape
        prefix_box = tuple()
        if have_raw_channels:
            if shape[-1] < 16:
                shape = shape[:-1]
            else:
                shape = shape[1:]
                prefix_box = (slice(None), )

        bb = self._sample_bounding_box(shape)
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
                    raise RuntimeError(
                        f"Could not sample a valid batch in {self.max_sampling_attempts} attempts"
                    )

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

        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels


    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)

        if self.label_transform2 is not None:
            labels = ensure_spatial_array(labels, self.ndim, dtype=initial_label_dtype)
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels


def create_data_loaders(train_set, val_set, test_set, batch_size, n_workers):
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        train_set, val_set, test_set: Dataset objects for training, validation, and testing.
        batch_size (int): Number of samples per batch.
        n_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: DataLoader objects for training, validation, and testing.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    train_loader.shuffle = True
    val_loader.shuffle = True
    test_loader.shuffle = True

    return train_loader, val_loader, test_loader


def StaccDataLoader(train_dataset_file, patch_shape, n_workers, batch_size, eps=None, sigma=None, lower_bound=None, upper_bound=None):
    train_images, train_labels, val_images, val_labels, test_images, test_labels = _split_data_paths_into_training_dataset(train_dataset_file)

    train_set = StaccImageCollectionDataset(train_images, train_labels, patch_shape, eps=eps, sigma=sigma,
                                            lower_bound=lower_bound, upper_bound=upper_bound)
    val_set = StaccImageCollectionDataset(val_images, val_labels, patch_shape, eps=eps, sigma=sigma,
                                          lower_bound=lower_bound, upper_bound=upper_bound)
    test_set = StaccImageCollectionDataset(test_images, test_labels, patch_shape, eps=eps, sigma=sigma,
                                           lower_bound=lower_bound, upper_bound=upper_bound)

    return create_data_loaders(train_set, val_set, test_set, batch_size, n_workers)


def StaccNapariDataLoader(path_to_data, n_workers, batch_size, average_object_width):
    images = glob.glob(os.path.join(path_to_data, "images", "*"))
    annotations = glob.glob(os.path.join(path_to_data, "annotations", "*"))

    train_images, train_labels, val_images, val_labels, test_images, test_labels = _split_csv_data(images, annotations)

    train_set = StaccImageCollectionDataset(train_images, train_labels, average_object_width=average_object_width)
    val_set = StaccImageCollectionDataset(val_images, val_labels, average_object_width=average_object_width)
    test_set = StaccImageCollectionDataset(test_images, test_labels, average_object_width=average_object_width)

    return create_data_loaders(train_set, val_set, test_set, batch_size, n_workers)

# def save_raw_and_label_visualization(image_path, csv_path, output_path, average_object_width=20, eps=0.00001):
#     """
#     Saves a visualization of the raw image and the corresponding STACC label created from a CSV file.

#     Args:
#         image_path (str): Path to the input raw image.
#         csv_path (str): Path to the CSV file containing point coordinates.
#         output_path (str): Path where the output image with visualization will be saved.
#         average_object_width (float): Average width of the objects to determine the Gaussian sigma.
#         eps (float, optional): Epsilon value for Gaussian truncation. Default is 0.00001.
#     """
#     # Load the raw image
#     raw_image = imread(image_path)

#     # Create the STACC label from the CSV
#     print("label creation started")
#     stacc_label = create_stacc_labels_from_csv(image_path, csv_path, average_object_width, eps)
#     print("label creation done.")
#     # Plot the raw image and the STACC label
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     ax1, ax2 = axes

#     ax1.imshow(raw_image, cmap='gray')
#     ax1.set_title('Raw Image')
#     ax1.axis('off')

#     ax2.imshow(stacc_label, cmap='hot')
#     ax2.set_title('STACC Label')
#     ax2.axis('off')

#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()


# if __name__ == "__main__":
#     # Example usage
#     print("example usage")
#     image_path = '/user/jeremias7/u12092/work/stacc/napari-example-data/images/image1.tif'
#     csv_path = '/user/jeremias7/u12092/work/stacc/napari-example-data/annotations/annotations1.csv'
#     output_path = 'output_visualization.png'
#     save_raw_and_label_visualization(image_path, csv_path, output_path)