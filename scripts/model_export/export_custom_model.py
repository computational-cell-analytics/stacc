import os
from stacc.training import export_model
import argparse


def main(model_path: str, output_folder: str) -> None:
    """
    Export a trained model to a specified output path to stacc-readable file.

    Args:
        model_path (str): Path to the trained model file.
        output_folder (str): Folder to save the exported model.

    Returns:
        None
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_path = os.path.join(output_folder, model_name + "_exported.pt")
    export_model(model_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained model to a specified output path.")
    parser.add_argument("-m", "--model", required=True, help="Path to the trained model file.")
    parser.add_argument("-o", "--output", required=True, help="Folder to save the exported model.")
    args = parser.parse_args()

    main(args.model, args.output)
