import argparse
import os
from PIL import ImageFile

# stacc package imports
from stacc.training import get_stacc_data_loader, load_config, run_stacc_training, export_model

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(args):
    """Main function to execute the training process.

    This function assumes label in COCO data format, e.g. from the AGAR dataset.
    The training configuration is stored in an extra training file, check out the file 'test_config.json'
    for an example configuration file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing the path to the configuration file.
    """
    config = load_config(args.config)

    train_loader, val_loader, _ = get_stacc_data_loader(
        config.train_dataset,
        n_workers=config.n_workers,
        eps=config.epsilon,
        batch_size=config.batch_size,
        patch_shape=config.patch_shape,
        sigma=config.sigma,
        lower_bound=config.lower_bound,
        upper_bound=config.upper_bound
    )

    run_stacc_training(
        config.model_name,
        train_loader,
        val_loader,
        config.learning_rate,
        device="cuda",
        pretrained_model_path=config.pretrained_model_path,
        save_new_model_path=config.save_new_model_path,
        iterations=config.iterations
    )

    #
    if args.export_path:
        checkpoint_path = os.path.join(config.save_new_model_path, "checkpoints", config.model_name, "best.pt")
        print("The trained model was exported to", args.export_path)
        export_model(checkpoint_path, args.export_path)


if __name__ == "__main__":
    # Create an argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train a UNet model for a counting task using a configuration file.")

    # Add an argument for the configuration file and parse it
    parser.add_argument(
        "config",
        type=str,
        help="Path to the JSON configuration file containing model and training parameters."
    )
    parser.add_argument(
        "--export_path",
        help="Path for exporting the trained model in a format that is compatible with the napri plugin."
    )
    args = parser.parse_args()

    main(args)
