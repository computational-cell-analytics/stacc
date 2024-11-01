# STACC

Automated counting for biomedical images.

This software includes the functionality to run prediction with our models,
either via the command line or with our napari plugin.
All of our code, including the training pipeline, will be documented and published
as open-source software upon publication of our manuscript.

We provide two different models:
- `cells`: A model for cell counting in phase-contrast microscopy. It was trained on the [LIVECell dataset](https://www.nature.com/articles/s41592-021-01249-6).
- `colonies`: A model for colony counting in images of culture medial. It was trained on the [AGAR dataset](https://www.researchsquare.com/article/rs-668667/v1).


## Installation

We recommend to install the software in a new conda environment following these steps:
```
$ conda create -n stacc -c pytorch -c conda-forge pytorch cpuonly scikit-image napari pyqt
$ conda activate stacc
$ pip install -e .
```

To install it in an existing python environment please ensure that the necessary environments (pytorch, scikit-image, napari) are installed in it and then run `$ pip install -e .` in that environment.


## Usage

We provide two options for using our counting method:

1. Via our napari plugin.

To start the plugin run `napari` in your environment. Then click `Plugins->Counting(Counting)`.
This will add the plugin widget to your napari viewer. You can use it to count cells or colonies by
selecting the respective model in the `model` dropdown, selecting the image you want to process in `image`
and then clicking `Count`.

The video `videos/example_usage.mp4` shows this in a short demo.

2. Via the command line.

You can use the command `stacc.counting` to count cells or colonies in image(s). Please run `$ stacc.counting -h` to see how the command is used.

We also provide two example images, one for cell counting and one for colony counting, in the folder `images`.
