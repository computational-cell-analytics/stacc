


"""
TODO whole file Tuedsday!
"""










import numpy as np
import torch
from os.path import basename
import json
import os
from skimage.io import imread
from skimage.filters import gaussian
from torch_em.util import (ensure_spatial_array, ensure_tensor_with_channels, load_image, supports_memmap)
from skimage.transform import resize

def apply_stamp_without_rand_and_bridges(stamp, position, label_matrix, image):
    """Adds a stamp (small matrix) at the coordinates 'position' of the label_matrix. """
    w_stamp, _ = stamp.shape # stamp has square shape, w_stamp is odd
    # print(f"width stamp: {w_stamp}")
    if w_stamp != _:
        print(f"Warning: stamp is not squared."
        f"Shape = {w_stamp} x {_} at coordinate {position}.")
    x, y = position
    try:
        h = w_stamp // 2
        lx, ly = label_matrix.shape
        if y+h+1 > ly or y-h < 0 or x+h+1 > lx or x-h < 0:
            # rechts
            if y+h+1 > ly:
                # rechts und unten
                if x+h+1 > lx:
                    # print("rechts und unten abgeschnitten")
                    a = label_matrix[x-h:lx, y-h:ly]
                    label_matrix[x-h:lx, y-h:ly] = np.maximum(a, stamp[:-abs((x+h+1) - lx), :-abs((y+h+1) - ly)])
                    return
                # rechts und oben
                elif x-h < 0:
                    # print("rechts und oben abgeschnitten")
                    a = label_matrix[0:x+h+1, y-h:ly]
                    label_matrix[0:x+h+1, y-h:ly] = np.maximum(a, stamp[abs(x-h):, :-abs((y+h+1) - ly)])
                    return
                else:
                    # print("rechts und mittig abgeschnitten")
                    a = label_matrix[x-h:x+h+1, y-h:ly]
                    label_matrix[x-h:x+h+1, y-h:ly] = np.maximum(a, stamp[:, :-abs((y+h+1) - ly)])
                    return

            # links
            if y-h < 0:
                # links und unten
                if x+h+1 > lx:
                    # print("links und unten abgeschnitten")
                    a = label_matrix[x-h:lx, 0:y+h+1]
                    label_matrix[x-h:lx, 0:y+h+1] = np.maximum(a, stamp[:-abs((x+h+1) - lx), abs(y-h):])
                    return
                # links und oben
                elif x-h < 0:
                    # print("links und oben abgeschnitten")
                    a = label_matrix[0:x+h+1, 0:y+h+1]
                    label_matrix[0:x+h+1, 0:y+h+1] = np.maximum(a, stamp[abs(x-h):, abs(y-h):])
                    return
                # links und mittig
                else:
                    # print("links und mittig abgeschnitten")
                    a = label_matrix[x-h:x+h+1, 0:y+h+1]
                    label_matrix[x-h:x+h+1, 0:y+h+1] = np.maximum(a, stamp[:, abs(y-h):])
                    return

            # nur oben
            if x-h < 0:
                # print("mittig und oben abgeschnitten")
                a = label_matrix[0:x+h+1, y-h:y+h+1]
                label_matrix[0:x+h+1, y-h:y+h+1] = np.maximum(a, stamp[abs(x-h):,:])
                return

            # nur unten
            if x+h+1 > lx:
                # print("mittig und unten abgeschnitten")
                a = label_matrix[x-h:lx, y-h:y+h+1]
                label_matrix[x-h:lx, y-h:y+h+1] = np.maximum(a, stamp[:-abs(x+h+1 - lx),:])
                return
        else:
            # print("nichts abgeschnitten")
            a = label_matrix[x-h:x+h+1, y-h:y+h+1]
            label_matrix[x-h:x+h+1, y-h:y+h+1] = np.maximum(a, stamp)
            return  
    except Exception as e:
        print(f"doch irgendwas schief gegangen in image {image}")
        print(f"Warning: Corresponding error message: {e}.")
    return

def width_to_sigma(width, eps, lower_bound, upper_bound):
    # shrink needs to be between 0 and 1
    sigma = np.sqrt(-(width**2) / (2*np.log(eps)))
    #### bounding ####
    if lower_bound and upper_bound:
        if sigma < lower_bound:
            sigma = lower_bound
        elif sigma > upper_bound:
            sigma = upper_bound
    # print(sigma)
    return int(sigma)

class JuliasException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

def create_gaussian_stamp(width, eps, lower_bound, upper_bound):
    """
    Creates a Gaussian stamp (matrix) with size width x width.
    If width is even, set width = width - 1.
    """
    if width % 2 == 0:
        width = width - 1
    
    sigma = width_to_sigma(width, eps, lower_bound, upper_bound)
    # put gaussian onto stamp
    #i, j = np.meshgrid(np.arange(-width//2+1, width//2+1), np.arange(-width//2+1, width//2+1))
    #stamp = np.exp(- (i**2 + j**2) / (2*sigma**2))

    stamp = np.zeros((width, width))
    stamp[width//2, width//2] = 1
    stamp = gaussian(stamp, sigma=sigma, truncate=10.0, mode='constant')
    # truncate s.t. stamp becomes circle:
    stamp[np.where(stamp < eps)] = 0
    # normalize s.t. center pixel = 1:
    #stamp = stamp / stamp[width//2, width//2]
    # alternatively, multiply 
    # eigentlich 2 * pi und so, aber ich mache 8, damit alles mal 4 genommen wird, sodass wir bei einem ähnlichen max wie damals rauskommen
    stamp = stamp * 8 * np.pi * sigma**2
    # print(f"Stamp Maximum: {stamp.max()}")
    return stamp

def resize_image(image):
    # hardgecoded, Erklärung siehe oben
    width = 2928
    resized = resize(image, (width, width))
    return resized

# create label matrix
def json_transform_to_matrix(json_label, eps=0.00001, sigma=None, lower_bound=None, upper_bound=None):
    with open(json_label) as label:
        label_dict = json.load(label)
        colonies = len(label_dict['labels'])
    
    number = basename(json_label)[:-5]

    # Check for both .jpg and .tif files
    image_path_jpg = json_label[:-5] + '.jpg'
    image_path_tif = json_label[:-5] + '.tif'

    if os.path.exists(image_path_jpg):
        image_path = image_path_jpg
    elif os.path.exists(image_path_tif):
        image_path = image_path_tif
    else:
        raise FileNotFoundError(f"Neither .jpg nor .tif image file found for {number}.")

    im = imread(image_path)
    rows = im.shape[0]
    columns = im.shape[1]

    labels = np.zeros((rows, columns))
    if colonies > 0:
        reasonable_indicies = [i for i in range(colonies) if (label_dict['labels'][i]['x'] + max(int(label_dict['labels'][i]['width']/2), 1)) < columns and (label_dict['labels'][i]['y'] + max(int(label_dict['labels'][i]['height']/2), 1)) < rows]
        if sigma: 
            x_coordinates = np.array([(int(label_dict['labels'][i]['x']) + max(int(label_dict['labels'][i]['width']/2), 1)) for i in reasonable_indicies], dtype='int')
            y_coordinates = np.array([(int(label_dict['labels'][i]['y']) + max(int(label_dict['labels'][i]['height']/2), 1)) for i in reasonable_indicies], dtype='int')
            labels[y_coordinates, x_coordinates] = 1
            labels = gaussian(labels, sigma=sigma, mode="constant")
            labels[np.where(labels < eps)] = 0
            labels = labels * 2 * 4 * np.pi * sigma**2 # *4 to normalize maxima to 4, as suggested by follow up paper of zisserman. *10 to get to 10, etc.
            return labels
        else:
            for i in reasonable_indicies:
                width = max(int(label_dict['labels'][i]['width']), 1)
                height = max(int(label_dict['labels'][i]['height']), 1)
                x_coord = int(label_dict['labels'][i]['x']) + width // 2
                y_coord = int(label_dict['labels'][i]['y']) + height // 2
                coords = (y_coord, x_coord)

                width = min(width, height)
                stamp = create_gaussian_stamp(width, eps, lower_bound, upper_bound)
                apply_stamp_without_rand_and_bridges(stamp, position=coords, label_matrix=labels, image=image_path)
            return labels
    else:
        return labels

class StaccImageCollectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500
    
    def _check_inputs(self, raw_images, label_images):
        if len(raw_images) != len(label_images):
            raise ValueError(f"Expect same number of raw and label images, got {len(raw_images)} and {len(label_images)}")

        # is_multichan = None
        for raw_im, label_im in zip(raw_images, label_images):
            # if basename(raw_im)[:-4] != basename(label_im)[:-4]:
            #     raise ValueError(f"Expect matching raw and label images, got {len(raw_images)} and {len(label_images)}")
            # we only check for compatible shapes if both images support memmap, because
            # we don't want to load everything into ram
            if supports_memmap(raw_im) and supports_memmap(label_im):
                shape = load_image(raw_im).shape
                assert len(shape) in (2, 3)

                multichan = len(shape) == 3
                if is_multichan is None:
                    is_multichan = multichan
                else:
                    assert is_multichan == multichan

                # we assume axis last
                if is_multichan:
                    # use heuristic to decide whether the data is stored in channel last or channel first order:
                    # if the last axis has a length smaller than 16 we assume that it's the channel axis,
                    # otherwise we assume it's a spatial axis and that the first axis is the channel axis.
                    if shape[-1] < 16:
                        shape = shape[:-1]
                    else:
                        shape = shape[1:]

                label = json_transform_to_matrix(label_im)
                # load_image(label_im).shape CHANGED
                label_shape = label.shape
                if shape != label_shape:
                    msg = f"Expect raw and labels of same shape, got {shape}, {label_shape} for {raw_im}, {label_im}"
                    raise ValueError(msg)

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
        eps=10**-8,
        sigma=None,
        lower_bound=None,
        upper_bound=None
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
        raw, label = self.raw_images[index], self.label_images[index]

        # print(f"Raw path: {raw}, label path: {label}")

        raw = load_image(raw)
        label = json_transform_to_matrix(json_label=label, eps=self.eps, sigma=self.sigma, lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        
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