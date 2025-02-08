STACC is a tool for automated counting in biomedical images.

It provides the following functionality:
- Counting objects in new images in a [napari plugin](#napari-plugin) or using the [command line](#command-line-interface).
- Two pretrained models for counting tasks:
    - `cells`: for cell counting in phase-contrast microscopy, trained on the [LIVECell dataset](https://www.nature.com/articles/s41592-021-01249-6).
    - `colonies`: for colony counting in images of culture medial, trained on the [AGAR dataset](https://www.researchsquare.com/article/rs-668667/v1).
- [Training](#training) a new counting model based on annotated data, with special support for data annotation in napari.
- A [python library](#python-library) to integrate our functionality within other tools.

The code, models and documentation will be made open-source upon publication of our manuscript and filing of a pending patent under a dual license that enables free use for academic purposes.


# Installation

We recommend to install the software in a new conda environment by opening a terminal and then following these steps:
```
git clone https://github.com/computational-cell-analytics/stacc
cd stacc
conda env create -f environment.yaml
conda activate stacc
pip install -e .
```

To install it in an existing python environment please ensure that the necessary dependencies (pytorch, scikit-image, napari) are installed in it and then run `$ pip install -e .` in that environment.

If you are not familiar with conda then check out [this introduction](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).


# Usage

The STACC tool provides a [napari plugin](#napari-plugin), a [command line interface](#command-line-interface), functionality for [training custom models](#training), and a [python library](#python-library).

We also provide two example images, one for cell counting and one for colony counting, in the folder `images`.
The foloder `images/example_training_data` contains (synthetic) example data for training a custom model.


## Napari Plugin 

The napari plugin supports counting on new images. To use it, start `napari` in your `stacc` environment, the click `Plugins->Counting(Counting)`. This will add the plugin to your napari viewer.
You can use it to count cells, colonies, or other objects (via a custom model you have trained) in images loaded in napari.
The user interface looks like this:
<img src="https://owncloud.gwdg.de/index.php/s/69Rc58gMpmsnL0s/download" width="768">

You can select the image via the `image` dropdown, which will show all images currently loaded in napari. The model dropdown offers the choice between the `colonies` and `cells` model.
A custom model (see [training](#training) for how to train your own model) can be loaded via `custom model path`.
The values for `min distance` and `threshold` determine the minimal distance between objects and the minimal intensity in the model predictions for detecting an object.
You usually don't have to change them if you are using a model on data that is similar to its training data, but changing the parameters may improve results if the data differs from training data.

Clicking on `Count` applies the model to the image. The result will be added as a [point layer](https://napari.org/0.4.15/howtos/layers/points.html) called `Counting Result` to napari, which displays each object that was found as a white dot on the image.
The number of objects is also displayed in the lower right corner.

The video `videos/example_usage.mp4` shows a short demo for using the napari plugin.


## Command Line Interface

The command line function `stacc.counting` can be used to count cells, colonies or other objects (with a custom model) in an image or in multiple images. For example, to segment the example cell image run
```bash
stacc.counting -i images/cell_example_image.png -m cells -o cell_counts
```
This will print the number of counted cells to the command line and will save the cell locations to a csv file in the output folder `cell_counts`.

```bash
stacc.counting -h
```
prints an overview of the full CLI functionality.


## Training

Custom models for counting can either be trained from data with annotations saved in [COCO style](https://cocodataset.org/#format-data) (e.g. the LIVECEll or AGAR datasets) or from annotations exported from napari point layers.
We provide two example scripts for this in `scripts/example_stacc_trainng.py` and `scripts/example_napari_training.py`. Training from napari labels is also supported via the CLI function `stacc.training_from_napari`.

Training from napari annotations enables correcting initial predictions from a STACC model to generate training data instead of having to annotate all objects in the training images by hand.
Here, we show this for a few images from a [dataset containing phase-contrast images](https://github.com/oist/Usiigaci).

The `cells` model yields ok results for this data, but is not yet perfect:
<img src="https://owncloud.gwdg.de/index.php/s/X6VL4MF5AmYtqTX/download" width="768">

We can correct the predictions by adding, removing or moving points, see [the point layer documentation](https://napari.org/0.4.15/howtos/layers/points.html) for how:
<img src="https://owncloud.gwdg.de/index.php/s/jD2QFXXynr979kF/download" width="768">

The point layer can then be saved by selecting it and clicking `File->Save Selected Layers ...`.
You should correct a few images this way, at least 10 if you can use a pre-trained model, more if you want to train from scratch.
Save the images and annotations (the saved point layers) in a folder structure like this:
```
images/
  image1.tif
  image2.tif
  image3.tif
  ...
annotations/
  points1.csv
  points2.csv
  points3.csv
  ...
```
You can then train a model on the data by running
```bash
stacc.training_from_napari -i images/ -l annotations/ -n my-cell-model -p cells -o 20
```
in the terminal. Here, `-i` and `-l` specify the path to the folder with images and annotations, respectively.
The argument `-n` determines the name of the model, `-p` specifies from which model to start the training.
Here, we use the `cells` model; if none is given the model will be trained from sratch.
The argument `-o` is used to specify the median object size, which you should determine from the data.
After the training finishes the model will be saved in the directory where you ran the script with the name `my-cell-model.pt` (or the value you have passed to `-n`).
You can then use this model for prediction in the [napari plugin](#napari-plugin) or in the [CLI](command-line-interface).

Even if none of the available counting models work for your data, this approach can be used to generate training data and train a custom model relatively quickly.


## Python Library

The `stacc` python library implements the functionality for prediction and training. Prediction with trained models is implemented in `stacc.prediction` and model training in `stacc.training`.

For example, to count cells in an image you can use it like this:
```python
import imageio.v3 as imageio
from stacc.prediction import run_counting
from stacc.util import get_model

image = imageio.imread("images/cell_example_image.png")
model = get_model("cells")
predicted_coordinates = run_counting(model, image)
```
