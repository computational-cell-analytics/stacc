"""This is an example script to train from point annotations generated with napari.
It uses the example annotations from 'images/example_training_data', which is organized in two sub-folders:
- images: containing the images as tif files.
- annotations: containing the point annotations in the csv format exported from napari.

If you save your data in the same format you can just change 'root_path' and run this script.
"""
import os
from stacc.training import run_stacc_training_from_napari_annotations

root_path = "../images/example_training_data"

run_stacc_training_from_napari_annotations(
    name="napari-example-model", pretrained_model_name="cells",  # Select the pretrained model, either cells or colonies.
    image_folder=os.path.join(root_path, "images"),
    label_folder=os.path.join(root_path, "annotations"),
    average_object_width=6,  # The average object size in pixels. You need to estimate this based on the data.
)
