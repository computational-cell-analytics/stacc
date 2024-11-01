from typing import Literal, TYPE_CHECKING
from magicgui import magic_factory
from napari.utils.notifications import show_info

from ..prediction import run_counting
from ..util import get_model

if TYPE_CHECKING:
    import napari


@magic_factory(call_button="Count")
def counting(
    image: "napari.layers.Image",
    model: Literal["colonies", "cells"] = "colonies",
    min_distance: int = 10,
    threshold_abs: float = 1.0,
) -> "napari.types.PointsData":
    model_ = get_model(model)
    image_data = image.data
    points = run_counting(model_, image_data, min_distance=min_distance, threshold_abs=threshold_abs)
    count = len(points)
    show_info(f"STACC counted {count} {model}.")
    return points
