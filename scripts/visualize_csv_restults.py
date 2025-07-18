import pandas as pd
from PIL import Image, ImageDraw
import argparse
import os
from pathlib import Path
import imageio
import numpy as np


def overlay_points_on_image(image: np.ndarray, points: np.ndarray, output_folder: str, image_name: str) -> None:
    """
    Overlay points onto an image and save the result.

    Args:
        image (np.ndarray): The image data as a NumPy array.
        points (np.ndarray): Array containing points to overlay.
        output_folder (str): Folder to save the output image with points overlaid.
        image_name (str): Name of the image file to construct the output path.

    Returns:
        None
    """
    # Convert the image to PIL format for drawing
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    # Overlay points on the image
    for y, x in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red", outline="red")  # Draw a small circle at each point

    # Construct the output file path using the image file name
    output_path = os.path.join(output_folder, f"{image_name}_results.jpg")

    # Save the output image
    img.save(output_path)


def main() -> None:
    """
    Parse command-line arguments, read image and points, and overlay points on the image.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Overlay points from a CSV file onto an image.")
    parser.add_argument("-i", "--image", help="Path to the input image.")
    parser.add_argument("-c", "--csv", help="Path to the CSV file containing points.")
    parser.add_argument("-o", "--output", help="Folder to save the output image with points overlaid.")
    args = parser.parse_args()

    # Ensure the output folder exists
    os.makedirs(args.output, exist_ok=True)

    # Read the image and CSV file
    image = imageio.imread(args.image)
    points = pd.read_csv(args.csv).to_numpy()
    image_name = Path(args.image).stem

    overlay_points_on_image(image, points, args.output, image_name)


if __name__ == "__main__":
    main()
