import pandas as pd
from argparse import ArgumentParser
import torch
from typing import Tuple
import numpy as np
import cv2
from hyptraining.utils.utils import Point
from hyptraining.datap.processing import (
    Circle,
    hypdataframe_to_tensor,
    find_circle,
    resize_image_to_screen,
    prompt_for_surrounding_points,
)


def get_arguments():
    ap = ArgumentParser()
    ap.add_argument(
        "--model_path",
        default="./model/weights.pth",
        help="Pretrained model location to use for visualization",
    )
    ap.add_argument(
        "--target_img",
        default="./trgt_imgs/img.parquet",
        help="Location of image to use for visualization.",
    )
    ap.add_argument("--image_height", default=1280, help="Hyperspectral image height.")
    ap.add_argument("--image_width", default=1024, help="Hyperspectral image width")

    return ap.parse_args()


def get_img_of_interest(img: np.ndarray) -> Tuple[np.ndarray, Circle]:

    gray_img = img[:, :, 60].squeeze() if img.shape[2] > 1 else img  # READ ONLY
    gray_img = np.mean(img, axis=-1).flatten()
    visual_rep = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2).astype(
        np.float32
    )  # Visual changes happen here'
    # Ask user to select within the imagek
    cv2.imshow("image", visual_rep)
    print("Please select three points to be used to find circle.")
    points_visual = prompt_for_surrounding_points("image")
    cv2.destroyWindow("image")
    visual_rep, scaling_factor = resize_image_to_screen(visual_rep)
    (
        circle_of_interest_coords_visual,
        circle_of_interest_radius_visual,
    ) = find_circle(*points_visual)

    circle_of_interest_coords_true = Point(
        int(circle_of_interest_coords_visual.x * (1 / scaling_factor)),
        int(circle_of_interest_coords_visual.y * (1 / scaling_factor)),
    )
    circle_of_interest_radius_true = int(
        circle_of_interest_radius_visual * (1 / scaling_factor)
    )

    #
    return


if __name__ == "__main__":
    args = get_arguments()
    # Load the model
    model = torch.load(args.model_path)
    # Load image
    df = pd.read_parquet(args.target_img)
    img = hypdataframe_to_tensor(df, args.image_height, args.image_width)

    # Select area of interest
    # Find Area of Interest
