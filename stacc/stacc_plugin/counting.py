from pathlib import Path
from typing import Optional

import napari
from magicgui import magic_factory
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel

from ..prediction import run_counting
from ..util import get_model, get_postprocessing_parameters


@magic_factory(
    model={"choices": ["colonies", "cells"]},
    call_button=False,
)
def _parameter_panel(
    model: str = "colonies",
    min_distance: int = get_postprocessing_parameters("colonies")[0],
    threshold: float = get_postprocessing_parameters("colonies")[1],
    custom_model_path: Optional[Path] = None,
):
    """This widget provides a control panel. It does not perform any computation by itself.
    """
    return


class CountingWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        layout = QVBoxLayout(self)

        # Add element to select an image.
        selector_widget = self._create_image_selector()
        layout.addWidget(selector_widget)

        # Add the other parameters.
        self.control = _parameter_panel()
        layout.addWidget(self.control.native)

        self.run_button = QPushButton("Count")
        layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.run_plugin)

        # Attach an event handler: when the model selection changes, update the post_processing_param value.
        self.control.model.changed.connect(self.update_post_processing)

    def _create_image_selector(self):
        self.layer_selectors = {}
        selector_widget = QWidget()
        image_selector = QComboBox()
        layer_label = QLabel("image")

        # Populate initial options
        layer_filter = napari.layers.Image
        self._update_selector(selector=image_selector, layer_filter=layer_filter)

        # Update selector on layer events
        self.viewer.layers.events.inserted.connect(lambda event: self._update_selector(image_selector, layer_filter))
        self.viewer.layers.events.removed.connect(lambda event: self._update_selector(image_selector, layer_filter))

        # Store the selector in the dictionary
        self.layer_selectors["image"] = selector_widget

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(layer_label)
        layout.addWidget(image_selector)
        selector_widget.setLayout(layout)
        return selector_widget

    def _update_selector(self, selector, layer_filter):
        """Update a single selector with the current image layers in the viewer."""
        selector.clear()
        image_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, layer_filter)]
        selector.addItems(image_layers)

    def update_post_processing(self, event):
        selected_model = event
        min_distance, threshold = get_postprocessing_parameters(selected_model)
        self.control.min_distance.value = min_distance
        self.control.threshold.value = threshold

    def run_plugin(self):
        """Called when the run button is pressed. Creates (or updates) a layer using the current parameter value."""
        # Get the current (possibly updated) post_processing_param from the control panel.
        min_distance = self.control.min_distance.value
        threshold = self.control.threshold.value

        # Get the model.
        model_ = get_model(self.control.model.value)

        # Get the image data.
        selector_widget = self.layer_selectors["image"]
        image_selector = selector_widget.layout().itemAt(1).widget()
        selected_layer_name = image_selector.currentText()
        image_data = self.viewer.layers[selected_layer_name].data

        # Run counting.
        points = run_counting(model_, image_data, min_distance=min_distance, threshold_abs=threshold)
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

        show_info(f"STACC counted {count} {self.control.model.value}.")
        self.viewer.add_points(points, **layer_kwargs)


# This is the plugin entrypoint that napari will call (via the Plugins menu)
def counting_plugin(viewer: napari.Viewer):
    return CountingWidget(viewer)
