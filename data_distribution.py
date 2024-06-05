import matplotlib.pyplot as plt
import argparse
import numpy as np
from torch import nn
import torch
import pandas as pd
import os, sys
from hyptraining.datap.processing import hypdataframe_to_tensor, thickess_to_img_space
from hyptraining.utils.utils import create_logger


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target_path",
        default="./data/raw/target_true.parquet",
        help="Location of saved model in format *.pth",
    )
    # Start exclusive argument group for either single image or folder of images
    ap.add_argument(
        "--image_path",
        help="Path to a single image to evaluate",
    )
    ap.add_argument("--image_height", default=1280, help="Hyperspectral image height.")
    ap.add_argument("--image_width", default=1024, help="Hyperspectral image width")
    return ap.parse_args()


if __name__ == "__main__":
    # Load Arguments
    args = get_args()
    logger = create_logger("MAIN")

    # Simple load up the data and visualize it into "2x1 plot"
    # Load the image
    logger.info("Loading data...")
    target_parquet = pd.read_parquet(args.target_path)

    features_path = args.target_path.replace("target", "features")
    features_parquet = pd.read_parquet(features_path)
    logger.info(f"Loaded target files with {len(target_parquet)} samples")

    logger.info("Processing data...")
    img_tensor = hypdataframe_to_tensor(
        features_parquet, args.image_width, args.image_height
    )
    thickness_img = thickess_to_img_space(target_parquet)
    # Make it into thee channels
    # img_averaged = img_tensor.mean(axis=-1)
    # img_final = np.stack((img_averaged,) * 3, axis=2)
    img_final = img_tensor[:, :, 60]

    # Matpplotlib
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    axs[0].imshow(img_final)
    axs[0].set_title("Hyperspectral Image")

    axs[1].imshow(thickness_img)
    axs[1].set_title(f"Thickness shown")

    plt.show()
