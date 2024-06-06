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
    # Ensure model is in evaluation
    model.eval()
    # parquet image
    features_parquet = pd.read_parquet(image_path)

    # Get the standard source
    final_img, ignore_spot = get_standard_source(
        features_parquet, template_loc, image_width, image_height, desired_angle
    )
    finimg_height, finimg_width, finimg_chan = final_img.shape
    logger.info(f"Final image shape {final_img.shape}")

    # Process the features
    feature_tensor = torch.from_numpy(final_img.reshape(-1, finimg_chan)).to(
        torch.float32
    )
    logger.info(f"Feature tensor is of shape {feature_tensor.shape}")
    inference = model(feature_tensor)
    logger.info(f"Inference is of shape {inference.shape}")
    inference_img = inference.reshape((finimg_height, finimg_width)).detach().numpy()
    logger.info(f"Reshaped inference is of shape {inference_img.shape}")

    # Crate ellipse mask:
    yslice, xslice = np.ogrid[:finimg_height, :finimg_width]
    vert_ellip_term = (yslice - finimg_height / 2) / (finimg_height / 2)
    hori_ellip_term = (xslice - finimg_width / 2) / (finimg_width / 2)

    # distances = np.sqrt(vert_ellip_term**2 + hori_ellip_term**2)

    mask = (vert_ellip_term**2 + hori_ellip_term**2) <= 1
    inference_img[~mask] = 0

    return inference_img


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
