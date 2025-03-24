import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import imageio.v3 as imageio
import numpy as np
import torch
from torch_em.util.modelzoo import export_bioimageio_model
from torch_em.transform.raw import standardize


def export_model(checkpoint_path: str, export_path: str) -> None:
    """Export a trained model from a checkpoint.

    The exported model can then be used within the napari plugin or the CLI for counting.

    Args:
        checkpoint_path: The path to the checkpoint.
        export_path: Where to save the model.
    """
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = model["model_state"]
    model_kwargs = model["init"]["model_kwargs"]
    print("The model from", checkpoint_path, "was created with the following kwargs:")
    print(model_kwargs)
    torch.save(model_state, export_path)


def export_bioimageio(
    checkpoint_path: str,
    output_path: str,
    sample_data: Union[str, np.ndarray],
    name: str,
    description: Optional[str] = None,
    authors: Optional[List[Dict]] = None,
    additional_citations: Optional[List[Dict]] = None,
    additional_tags: Optional[List[str]] = None,
    documentation: Optional[str] = None,
) -> None:
    """Export a trained model checkpoint as bioimage.io model.

    Args:
        checkpoint path: The checkpoint of the trained model.
        output_path: The path for saving the exported model.
        sample_data: The path to the input sample / test data.
            This has to be an image file that can be loaded by imageio or a numpy array.
        name: The name of the model.
        description: The description of this model. If None, a standard description for the STACC model will be used.
        authors: The developers of this model. If None, the STACC core developers will be listed.
        additional_citations: Additional citations for this model. If None, only the STACC publication will be listed.
        additional_tags: Additional tags for this model.
        documentation: The documentation for this model. By default a general documentation of STACC will be used.
    """
    if isinstance(sample_data, str):
        sample_data = imageio.imread(sample_data)
        sample_data = standardize(sample_data)

    if authors is None:
        authors = [
            {"name": "Julia Jeremias", "github_user": "julia-jeremias"},
            {"name": "Constantin Pape", "github_user": "constantinpape"}
        ]

    tags = ["unet2d", "pytorch", "counting"]
    if additional_tags is not None:
        tags += additional_tags

    cite = [
        {"text": "Pape and Jeremias", "doi": "10.1515/mim-2024-0021"}
    ]
    if additional_citations is not None:
        cite += additional_citations

    if description is None:
        description = "STACC is a model for detection and counting of objects in images or micrographs."

    # TODO this is just a placeholder, enter the correct license later
    license = "MIT"

    if documentation is None:
        documentation = "https://github.com/computational-cell-analytics/stacc/tree/main/scripts/model_export/documentation.md"  # noqa
    git_repo = "https://github.com/computational-cell-analytics/stacc"

    # TODO STACC config (= best values for find maxima)
    config = {}

    export_bioimageio_model(
        checkpoint=checkpoint_path,
        output_path=output_path,
        input_data=sample_data,
        name=name,
        description=description,
        authors=authors,
        tags=tags,
        license=license,
        documentation=documentation,
        git_repo=git_repo,
        cite=cite,
        input_optional_parameters=False,
        for_deepimagej=True,
        maintainers=authors,
        config=config,
    )


@dataclass
class TrainingConfig:
    """Data class to store training configuration parameters.

    Attributes:
        model_name: The name of the model.
        train_dataset: Path to the training dataset file.
            Should be a json, containing a dictionary with train, val, and test image and corresponding label paths.
        pretrained_model_path: Path to the pretrained model checkpoint.
        save_new_model_path: Path to save the new model checkpoints. By default, latest and best will be stored.
        batch_size: Batch size for the data loader.
        patch_shape: Shape of the patches for training.
        n_workers: Number of workers for data loading.
        iterations: Number of iterations for training.
        n_epochs: Number of epochs for training.
        learning_rate: Learning rate for the optimizer.
        epsilon: Truncate value for Gaussian stamp in STACC.
        sigma: Sigma parameter for STACC. Determines the size of the Gaussian stamp.
        lower_bound: Lower bound for STACC. Lower bound for Gaussian stamp size.
        upper_bound: Upper bound for STACC. Upper bound for Gaussian stamp size.
        augmentations: List of augmentations to apply.
        comment: Additional training description
    """
    model_name: str
    train_dataset: str
    test_dataset: str
    pretrained_model_path: str
    save_new_model_path: str
    batch_size: int
    patch_shape: Tuple[int, int]
    n_workers: int
    iterations: int
    n_epochs: int
    learning_rate: float
    epsilon: float
    sigma: float
    lower_bound: float
    upper_bound: float
    augmentations: List[str]
    comment: str


def load_config(config_file_path: str) -> TrainingConfig:
    """Load training configuration from a JSON file.

    Args:
        config_file_path: Path to the JSON configuration file.

    Returns:
        An instance of TrainingConfig with parameters loaded from the file.
    """
    with open(config_file_path) as file:
        parameters_dict = json.load(file)
    return TrainingConfig(**parameters_dict)
