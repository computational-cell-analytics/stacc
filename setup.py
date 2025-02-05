import runpy
from setuptools import setup, find_packages

__version__ = runpy.run_path("stacc/__version__.py")["__version__"]


setup(
    name="stacc",
    packages=find_packages(exclude=["test"]),
    version=__version__,
    author="Julia Jeremias; Constantin Pape",
    url="https://github.com/computational-cell-analytics/stacc",
    license="",
    entry_points={
        "console_scripts": [
            "stacc.counting = stacc.prediction:main",
        ],
        "napari.manifest": [
            "stacc = stacc:napari.yaml",
        ],
    },
)
