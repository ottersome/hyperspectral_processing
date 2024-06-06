import matplotlib.pyplot as plt
import argparse
import numpy as np
from numpy import pi
from torch import nn
from model import Model
import pandas as pd
import torch
import os, sys
from hyptraining.datap.processing import get_standard_source, hypdataframe_to_tensor
from hyptraining.utils.utils import Point, create_logger, draw_point_in_image


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        default="./models/weights.pth",
        help="Location of saved model in format *.pth",
    )
    # Start exclusive argument group for either single image or folder of images
    ap.add_argument(
        "--image_path",
        default="./data/raw/features_true.parquet",
        help="Path to a single image to evaluate. In parquet format",
    )
    ap.add_argument("--image_height", default=1280, help="Hyperspectral image height.")
    ap.add_argument("--image_width", default=1024, help="Hyperspectral image width")
    ap.add_argument(
        "--image_channels", default=120, type=int, help="Size of model's input"
    )
    ap.add_argument(
        "--desired_angle", default=pi / 2, help="Angle we want to orientate to"
    )
    ap.add_argument(
        "--template_location",
        default="./feature_right.png",
        help="Template for detecting features",
    )

    return ap.parse_args()


def single_image_comparison(model: nn.Module, img_path: str):
    """
    Will return a plot to show the image process
    """
    img: np.ndarray = plt.imread(img_path)
    # Sample all hyperspectral points from image and pipe them through the model
    output_img = np.zeros_like(img)
    for i, hyperspectral_point in enumerate(img):
        output_img[i] = model(hyperspectral_point)

    # Create the plot without showin


def multiple_image_comparison(model: nn.Module, folder_output_path: str):
    """
    Will save all recieved plots to the folder
    """
    pass


def single_image_view(
    model: nn.Module,
    image_path: str,
    template_loc: str,
    image_width: int,
    image_height: int,
    desired_angle: float,
) -> np.ndarray:
    # parquet image
    features_parquet = pd.read_parquet(image_path)

    # Get the standard source
    final_img, ignore_spot = get_standard_source(
        features_parquet, template_loc, image_width, image_height, desired_angle
    )
    img_center = Point(final_img.shape[1] // 2, final_img.shape[0] // 2)
    logger.debug(f"Now going into the debugging section")
    predicted_image = np.zeros((final_img.shape[0], final_img.shape[1]))

    # Finally this image is passed to the angle
    cache = []
    for i in range(final_img.shape[0]):
        for j in range(final_img.shape[1]):
            # Skip outside of circle
            ver_ell_term = (i - final_img.shape[0] / 2) / (final_img.shape[0] / 2)
            hor_ell_term = (j - final_img.shape[1] / 2) / (final_img.shape[1] / 2)
            # Ensure its within the ellipses
            if (hor_ell_term**2 + ver_ell_term**2) > 1:
                continue
            features = final_img[i, j, :]
            assert (
                len(features) == 120
            ), f"Length of features obtaines is not expected 120 but rather {len(features)}"
            features = torch.from_numpy(features).to(torch.float32)

            infered_thickness = model(features).detach().numpy().item()
            predicted_image[i, j] = infered_thickness

    return predicted_image


if __name__ == "__main__":
    # Load Arguments
    args = get_args()
    logger = create_logger("MAIN")

    logger.info(f"Importing model {args.model_path}")
    model = Model(args.image_channels, 1)
    model.state_dict = torch.load(args.model_path)
    model.eval()

    # Just work with single image for now
    predicted_image = single_image_view(
        model,
        args.image_path,
        args.template_location,
        args.image_width,
        args.image_height,
        args.desired_angle,
    )
    # Plot the predicted image
    plt.imshow(predicted_image)
    plt.show()
