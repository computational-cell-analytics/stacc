import random
import json
import itertools
import pandas
import warnings
import numpy as np
import torch

from tqdm import tqdm
from torch_em.util import load_image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from .prediction import run_counting
from .util import get_model
from .training.util import TrainingConfig


def _get_test_data(data):
    return data["test"]["images"], data["test"]["labels"]


def _get_center_coordinates(json_path, return_cell_types=False):
    with open(json_path, "r") as file:
        data = json.load(file)

    center_coordinates = []
    cell_types = []
    for label in data["labels"]:

        x = label["x"]
        y = label["y"]
        width = label["width"]
        height = label["height"]
        center_coordinates.append([y, x])
        cell_types.append(label["class"])

    center_coordinates = np.array(center_coordinates)

    if return_cell_types:
        return center_coordinates, cell_types
    return center_coordinates


def _calc_sMAPE(n, m):
    if n == 0 and m == 0:
        return 0
    elif (n == 0 and m > 0) or (n > 0 and m == 0):
        return 1
    elif n > 0 and m > 0:
        # from https://arxiv.org/pdf/2108.01234.pdf
        return abs(n - m) / (abs(n) + abs(m))
    else:
        return 1


def _calc_mae(n, m):
    return np.abs(n - m)


def _compute_pairwise_distances(gt, pred):
    pairwise_distances = cdist(gt, pred, metric="euclidean")

    if np.any(pairwise_distances):
        pairwise_distances = np.stack(pairwise_distances)
    return pairwise_distances


def _compute_performance_metrics(n, m, tp):
    # compute false positives and false negatives
    false_positives = m - tp
    false_negatives = n - tp

    precision = tp / (tp + false_positives) if tp > 0 else 0
    recall = tp / (tp + false_negatives) if tp > 0 else 0
    # from https://www.v7labs.com/blog/f1-score-guide#:~:text=The%20F1%20score%20is%20calculated,denotes%20a%20better%20quality%20classifier.
    f1 = (2 * precision * recall) / (precision + recall) if (precision > 0 and recall > 0) else 0
    smape = _calc_sMAPE(n, m)
    mae = _calc_mae(n, m)
    return precision, recall, f1, smape, mae


def _metric_coords(gts, preds, match_distance=45):
    """Computes Precision, Recall, F1, sMAPE, and MAE.

    Args:
        gts nd.array of tuples: Ground truth objects center coordinates.
        preds nd.array of tuples: Predicted center coordinates.
        match_distance (int, optional): Maximum distance to match two points as TP. Defaults to 45.

    Returns:
        precision, recall, f1, smape, mae
    """
    n = len(gts)
    m = len(preds)

    if n == 0 and m == 0:
        return 1, 1, 1, 0, 0

    elif (n == 0 and m > 0) or (n > 0 and m == 0):
        return 0, 0, 0, 1, _calc_mae(n, m)

    elif n > 0 and m > 0:
        pairwise_distances = _compute_pairwise_distances(gts, preds)
        if np.any(pairwise_distances):
            trivial = not np.any(pairwise_distances < match_distance)
            if trivial:
                true_positives = 0
            else:
                # match the predicted points to labels via linear cost assignment ('hungarian matching')
                max_distance = pairwise_distances.max()
                # the costs for matching: the first term sets a low cost for all pairs that are in
                # the matching distance, the second term sets a lower cost for shorter distances,
                # so that the closest points are matched
                costs = (
                    -(pairwise_distances < match_distance).astype(float)
                    - (max_distance - pairwise_distances) / max_distance
                )
                # perform the matching, returns the indices of the matched coordinates
                label_ind, pred_ind = linear_sum_assignment(costs)
                # check how many of the matches are smaller than the match distance
                # these are the true positives
                match_ok = pairwise_distances[label_ind, pred_ind] < match_distance
                true_positives = np.count_nonzero(match_ok)

            return _compute_performance_metrics(n, m, true_positives)
        else:
            # TODO: correct this one, as np.any(...) = False can mean all coordinates are exactly equal
            warnings.warn(
                f"Case where m = n = 1 should be true. Sanity check: n = {n}, m = {m}. "
                f"Also, coords should be equal: ground truth = {gts}, predictions = {preds}"
            )
            return 0, 0, 0, _calc_sMAPE(n, m), _calc_mae(n, m)
    else:
        raise Exception(
            f"Number of ground truths and/or predictions are negative (metric): " f"len(gts) = {n}, len(preds) = {m}."
        )


def _get_distance_threshold_from_gridsearch(data_frame, distances, threshes):
    combine = list(itertools.product(distances, threshes))
    mean_score = 0
    dist = 0
    thresh = 0.0

    for c in combine:
        mask = (data_frame["Distance"] == c[0]) & (data_frame["Threshold"] == c[1])
        param_kombi_df = data_frame[mask]

        meany = param_kombi_df.mean(axis=0).iloc[0]

        if mean_score < meany:
            mean_score = meany
            dist = c[0]
            thresh = c[1]

    return dist, thresh


def gridsearch_find_distance_and_threshold(config: TrainingConfig, model: torch.nn.Module, percentage: float = 0.1):
    """Find optimal postprocessing parameters **after** training.

    This function performs a gridsearch to find the optimal post-processing parameters for a trained model.
    It uses a subset of test images to evaluate different combinations of distance and threshold values.

    Args:
        config: Configuration file containing paths to the model and datasets used for training/testing.
                The dataset file should contain paths to the images and labels.
        model: The trained model to be used for counting. May be a custom trained model, exported after training.
        percentage (optional): Defines the proportion of images from the test set to be used for gridsearch.
                                      Defaults to 0.1.

    Returns:
        tuple: A tuple containing:
            - List of images used for gridsearch.
            - Optimal distance value and
            - Optimal threshold value for peak_local_max (skimage).
    """
    with open(config.test_dataset) as test_dataset:
        test_data = json.load(test_dataset)
    test_images, test_labels = _get_test_data(test_data)

    n = len(test_images)
    n_gridserach = int(percentage * n)
    random_gridsearch_image_idx = random.sample(range(n), n_gridserach)
    images_used_for_gridsearch = [test_images[idx] for idx in random_gridsearch_image_idx]

    distances = [2, 5, 10, 15, 20, 25]
    thresholds = np.arange(0.8, 2.6, 0.1).tolist()  # no numerical operations to come

    gridsearch_data = []
    for idx in tqdm(random_gridsearch_image_idx):
        image_path = test_images[idx]
        label_path = test_labels[idx]

        image_data = load_image(image_path)  # ndimage
        label_coords = _get_center_coordinates(label_path, return_cell_types=False)

        for dist, thresh in tqdm(
            itertools.product(distances, thresholds)
        ):  # heavy! you may want change distances and thresholds manually.
            pred_coords = run_counting(model, image_data, min_distance=dist, threshold_abs=thresh)
            (
                _,
                _,
                f1,
                _,
                _,
            ) = _metric_coords(
                label_coords, pred_coords
            )  # you may adjust this if you want to optimize it for a different metric
            # print(f"pr: {precision}, re: {recall}, f1: {f1}, smape: {smape}, mae: {mae}")
            gridsearch_data.append([f1, dist, thresh])

    df = pandas.DataFrame(data=gridsearch_data, columns=["f1", "Distance", "Threshold"])
    dist, thresh = _get_distance_threshold_from_gridsearch(df, distances, thresholds)

    return images_used_for_gridsearch, dist, thresh


def main():
    """@private

    Command-line interface for running gridsearch.

    This function sets up the command-line interface for running the gridsearch to find optimal post-processing parameters.
    It parses command-line arguments and executes the gridsearch process.

    Command-line Arguments:
        -c, --config: The filepath to your training config file.
        -cm, --custom_model: The filepath to your trained and exported custom model.
        -m, --model: The model to use for counting. Can either be 'colonies' for colony counting or 'cells' for cell counting.
                     By default, the model for colony counting is used.
        -o, --output: The output folder for saving results (JSON containing images used for gridsearch, distance, and threshold).

    Example:
        Run the gridsearch with the following command:
        stacc.gridsearch -c path/to/config -cm path/to/exported/model -m cells -o path/to/tmp
    """
    import argparse
    import os
    from .training import load_config

    parser = argparse.ArgumentParser(
        description="Run gridsearch to find optimal post processing parameters. "
        "For example, you can run 'stacc.gridsearch -c path/to/config -m path/to/exported/model -m cells -p 0.05 -o path/to/tmp"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="The filepath to your training config file.",
    )
    parser.add_argument(
        "-cm",
        "--custom_model",
        required=True,
        help="The filepath to your trained **and exported** custom model.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="colonies",
        help="The model to use for counting. Can either be 'colonies' for colony counting or 'cells' for cell counting."
        " By default the model for colony counting is used.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="The output folder for saving results (json containing images used for gridsearch, distance, and threshold).",
    )

    args = parser.parse_args()

    print(f"Starting gridsearch for {args.model}.")

    model = get_model(args.model, args.custom_model)
    config = load_config(args.config)
    images, distance, threshold = gridsearch_find_distance_and_threshold(config, model, percentage=0.05)

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, f"{config.model_name}_gridsearch_results.json")

        json_data = {"images_used_in_gridsearch": images, "distance": distance, "threshold": threshold}

        with open(out_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
