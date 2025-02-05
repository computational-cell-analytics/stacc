import argparse
from PIL import ImageFile

# stacc package imports
from stacc import StaccNapariDataLoader
from stacc import run_stacc_training

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(args):
    """
    Main function to execute the training process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing the path to the configuration file.
    """
    data_path = args.path_to_data
    average_object_size = args.object_size
    n_workers = args.n_workers
    batch_size = args.batch_size

    train_loader, val_loader, _ = StaccNapariDataLoader(data_path,
                                                        n_workers=n_workers,
                                                        batch_size=batch_size,
                                                        average_object_width=average_object_size
                                                        )

    run_stacc_training(args.name,
                       train_loader,
                       val_loader,
                       args.learning_rate,
                       device="cuda",
                       pretrained_model_path=args.pretrained,
                       save_new_model_path=args.save_path,
                       iterations=args.iterations
                       )


if __name__ == "__main__":
    # Create an argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train a UNet model for a counting task based on Napari annotations.")

    # Add an argument for the configuration file and parse it
    parser.add_argument(
        "path_to_data",
        type=str,
        help=(
            "Path to napari data, containing images/ and annotations/ with matching "
            "file names, except for extensions."
        )
    )

    parser.add_argument("--object_size", type=int, default=20,
                        help="Size of the objects to be considered in the model. Default is 20."
                        )

    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for the DataLoader. Default is 2."
                        )

    parser.add_argument("--n_workers", type=int, default=6, 
                        help="Number workers for the Dataloader. Default is 6."
                        )
    # @Constantin: passt zeitlich nicht mehr von meiner Seite aus.
    # parser.add_argument(
    #     "--pretrained",
    #     type=str,
    #     default="colonies",
    #     help="Defnies which checkpoints to use for fine-tuning. Either 'cells' or 'colonies'. Default is 'colonies'."
    # )

    parser.add_argument("--pretrained", type=str,
                        default="/scratch-emmy/usr/nimjjere/models/checkpoints/stacc_6_9_pretrained_on_agar_combined/best.pt",
                        help="Path to pretrained model for loading checkpoints. Default is None."
                        )

    parser.add_argument("--name", type=str, default="from_napari",
                        help="Model name for storing checkpoints. Default is 'from_napari'"
                        )

    parser.add_argument("--iterations", type=int, default=10000,
                        help="Number of training iterations. Default set to 10000."
                        )

    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training. Default 1e-4."
                        )

    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to store trained model."
                        )

    args = parser.parse_args()

    main(args)
