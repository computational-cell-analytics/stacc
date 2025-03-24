from stacc.training.util import export_bioimageio

agar_citation = {
    "text": "Edlund et al.", "doi": "10.1038/s41592-021-01249-6"
}
sample_data = "../../images/cell_example_image.png"
export_bioimageio(
    checkpoint_path="./checkpoints",  # TODO
    output_path="./stacc-cell-counting.zip",
    sample_data=sample_data,
    name="stacc-cell-counting",
    description="",  # TODO
    additional_citations=[agar_citation],
    additional_tags=["cells", "phase-contrast-microscopy"],
    documentation="./documentation_livecell.md",
)
