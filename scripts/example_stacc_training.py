import argparse
from PIL import ImageFile

# stacc package imports
from stacc import StaccDataLoader
from stacc import load_config
from stacc import run_stacc_training

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(args):
    """
    Main function to execute the training process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing the path to the configuration file.
    """
    config = load_config(args.config)

    train_loader, val_loader, _ = StaccDataLoader(
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


if __name__ == "__main__":
    # Create an argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train a UNet model for a counting task using a configuration file.")

    # Add an argument for the configuration file and parse it
    parser.add_argument(
        "config", 
        type=str, 
        help="Path to the JSON configuration file containing model and training parameters."
    )
    args = parser.parse_args()

    main(args)
