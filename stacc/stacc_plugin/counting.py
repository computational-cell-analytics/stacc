from typing import Literal, TYPE_CHECKING
from magicgui import magic_factory
from napari.utils.notifications import show_info

from ..prediction import run_counting
from ..util import get_model, get_postprocessing_parameters

if TYPE_CHECKING:
    import napari


@magic_factory(call_button="Count")
def counting(
    image: "napari.layers.Image",
    model: Literal["colonies", "cells"] = "colonies",
    # min_distance: int = 10,
    # threshold_abs: float = 1.0,
) -> "napari.types.LayerDataTuple":

    # Get the model and postprocessing settings.
    model_ = get_model(model)
    min_distance, threshold_abs = get_postprocessing_parameters(model)

    # Run counting.
    image_data = image.data
    points = run_counting(model_, image_data, min_distance=min_distance, threshold_abs=threshold_abs)
    count = len(points)

    # Set the size of the points dependend on the size of the image.
    image_shape = image_data.shape if image_data.ndim == 2 else image_data.shape[:-1]
    if any(sh > 2048 for sh in image_shape):
        point_size = 20
    else:
        point_size = 10
    layer_kwargs = {
        "name": "Counting Result",
        "size": point_size,
    }

    show_info(f"STACC counted {count} {model}.")
    return points, layer_kwargs, "points"
