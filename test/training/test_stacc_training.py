import os
import unittest
from shutil import rmtree

import imageio.v3 as imageio
import pandas as pd
from skimage.data import binary_blobs
from skimage.measure import label, regionprops


class TestStaccTraining(unittest.TestCase):
    tmp_folder = "./tmp"

    def _generate_training_sample(self, image_folder, label_folder, i):
        image = binary_blobs(128, blob_size_fraction=0.1, volume_fraction=0.1)
        labels = label(image)

        props = regionprops(labels)
        index = [prop.label for prop in props]
        ax0 = [prop.centroid[0] for prop in props]
        ax1 = [prop.centroid[1] for prop in props]
        labels = pd.DataFrame({"index": index, "axis-0": ax0, "axis-1": ax1})

        image_path = os.path.join(image_folder, f"image_{i}.tif")
        label_path = os.path.join(image_folder, f"labels_{i}.csv")

        imageio.imwrite(image_path, image)
        labels.to_csv(label_path, index=False)

        return image_path, label_path

    # Create sample training data in the napari data format.
    def _create_napari_training_data(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

        data_dict = {
            "train": {"images": [], "labels": []},
            "val": {"images": [], "labels": []},
        }

        image_folder = os.path.join(self.tmp_folder, "images")
        os.makedirs(image_folder, exist_ok=True)

        label_folder = os.path.join(self.tmp_folder, "labels")
        os.makedirs(label_folder, exist_ok=True)

        n_samples = 5
        for i in range(n_samples):
            image_path, label_path = self._generate_training_sample(image_folder, label_folder, i)
            if i == (n_samples - 1):  # use last image for val:
                data_dict["val"]["images"].append(image_path)
                data_dict["val"]["labels"].append(label_path)
            else:
                data_dict["train"]["images"].append(image_path)
                data_dict["train"]["labels"].append(label_path)

        return data_dict

    # Remove the tmp folder with the training data and the trained model checkpoint.
    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_run_stacc_training_from_napari_labels(self):
        from stacc.training import run_stacc_training, get_stacc_data_loader, width_to_sigma

        train_dict = self._create_napari_training_data()

        sigma = width_to_sigma(6, lower_bound=None, upper_bound=None)  # assume the average object width is 6 pixels.
        patch_shape = (128, 128)
        train_loader, val_loader, _ = get_stacc_data_loader(
            train_dict, patch_shape, batch_size=1, n_workers=1, sigma=sigma
        )

        model_name = "test-model"
        run_stacc_training(
            model_name, train_loader, val_loader, n_epochs=3, save_new_model_path=self.tmp_folder
        )

        # Check that the model has been trained
        self.assertTrue(os.path.join(self.tmp_folder, "checkpoints", model_name, "best.pt"))
        self.assertTrue(os.path.join(self.tmp_folder, "checkpoints", model_name, "latest.pt"))


if __name__ == "__main__":
    unittest.main()
