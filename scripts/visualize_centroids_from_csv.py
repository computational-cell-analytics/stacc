import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os


def overlay_points_on_image(image_path, csv_path, output_folder):
    # Load the image
    image = Image.open(image_path)
    plt.imshow(image)

    # Load the CSV file
    data = pd.read_csv(csv_path)

    # Plot the points with smaller size
    plt.scatter(data["x"], data["y"], c="red", marker="o", s=10)  # 's' parameter controls the size

    # Save the visualization
    plt.axis("off")  # Hide axes

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract the base name of the image file and create the output file name
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file_name = f"{base_name}_results.png"
    output_path = os.path.join(output_folder, output_file_name)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay points from CSV onto an image.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    parser.add_argument("-c", "--csv", required=True, help="Path to the CSV file with coordinates")
    parser.add_argument("-o", "--output", required=True, help="Path to the output folder")

    args = parser.parse_args()

    overlay_points_on_image(args.image, args.csv, args.output)
