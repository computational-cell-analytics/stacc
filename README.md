# STACC

STACC is a tool for automated counting in biomedical images.

You can find its documentation in the file `stacc-documentation.pdf`.
This documentation is built with pdoc and will be hosted online as soon as our tool is published.
The source for the documentation can be found in `doc/documentation.md`

The documentation also contains an (automatically generated) API documentation of the python library, which is not included in the pdf (but will be part of the online documentation).
To build the documentation yourself, install `pdoc` via
```
pip install pdoc
```
and then run:
```bash
pdoc --docformat google stacc/
```
