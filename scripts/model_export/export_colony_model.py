from stacc.training.util import export_bioimageio

agar_citation = {
    "text": "Majchrowska et al.", "doi": "10.48550/arXiv.2108.01234"
}
sample_data = "../../images/colony_example_image.jpg"
export_bioimageio(
    checkpoint_path="",  # TODO
    output_path="./stacc-colony-counting.zip",
    sample_data=sample_data,
    name="stacc-colony-counting",
    description="",  # TODO
    additional_citations=[agar_citation],
    additional_tags=["colonies", "bacteria", "funghi"],
    documentation="./documentation_colonies.md",
)
