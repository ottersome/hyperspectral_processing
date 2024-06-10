import matplotlib.pyplot as plt
import argparse
import numpy as np
from numpy import pi
from typing import Any
import matplotlib.pyplot as plt
from torch import nn
from scipy.interpolate import Rbf
from model import Model
import pandas as pd
import torch
from hyptraining.datap.processing import get_standard_source
from hyptraining.utils.utils import Point, create_logger


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
        "--feature_image_size",
        default=150 * 2,
        help="Size of the image containing the radius thing",
    )
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


def image_inference(
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
    final_img, _ = get_standard_source(
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
    inference_img[~mask] = np.nan

    return inference_img


def rbf_interpolation(x: np.ndarray, y: np.ndarray, values, img_size: Point):
    """
    Will do the same as `rbf_interpolation` but will plot results to ensure that
    the intepolation looks correct
    """
    # Normalize the points according to image size
    img_halfsize = Point(img_size.x // 2, img_size.y // 2)
    x_scaled = (x - x.min()) / (x.max() - x.min()) * img_size.x
    y_scaled = (y - y.min()) / (y.max() - y.min()) * img_size.y

    # Generate the grid
    xi = np.arange(0, img_size.x)
    yi = np.arange(0, img_size.y)
    XI, YI = np.meshgrid(xi, yi)

    # Do RBF-Interpolation
    rbf = Rbf(x_scaled, y_scaled, values, function="multiquadric")
    ZI = rbf(XI, YI)

    # Crate the mask to remove non-disk pixels
    horterm = (XI - img_halfsize.x) / img_halfsize.x
    verterm = (YI - img_halfsize.y) / img_halfsize.y
    mask = horterm**2 + verterm**2 > 1
    ZI[mask] = np.nan

    # Sanity
    sparse_groundtruth = np.zeros(img_size)
    sparse_groundtruth[sparse_groundtruth == 0] = np.nan
    for i, j, z in zip(y_scaled, x_scaled, values):
        logger.debug(f"Inserting fpoint ({i},{j})")
        ii = min(int(i), img_size.y - 1)
        ij = min(int(j), img_size.x - 1)
        logger.debug(f"Inserting ipoint ({ii},{ij})")
        sparse_groundtruth[ii, ij] = z

    return ZI, sparse_groundtruth


def raw_dump(obj: Any, array_name: str, file: str):
    with open(file, "w+") as f:
        f.write(f"Start dump of {array_name}--------------------\n")
        f.write(obj)
        f.write(f"End dump of {array_name}-------------------\n")


if __name__ == "__main__":
    # Load Arguments
    args = get_args()
    logger = create_logger("MAIN")

    logger.info(f"Importing model {args.model_path}")
    model = Model(args.image_channels, 1)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Crate Interpolation of target image for visuals
    target_image = pd.read_parquet(args.image_path.replace("features", "target"))
    logger.info(f"Have loaded image with {len(target_image)} features")
    final_image_size = Point(args.feature_image_size, args.feature_image_size)
    interpolated_approx, sparse_groundtruth = rbf_interpolation(
        target_image.iloc[:, 0],
        target_image.iloc[:, 1],
        target_image.iloc[:, 2],
        final_image_size,
    )

    # Just work with single image for now
    predicted_image = image_inference(
        model,
        args.image_path,
        args.template_location,
        args.image_width,
        args.image_height,
        args.desired_angle,
    )

    # For the sake of uniform vmap
    imgs = [interpolated_approx, sparse_groundtruth]
    gmin = min([np.nanmin(i) for i in imgs])
    gmax = max([np.nanmax(i) for i in imgs])

    logger.info(
        f"Interpolated min:{interpolated_approx.min()} and max: {interpolated_approx.max()}"
    )
    logger.info(f"Gmin {gmin} gmax {gmax}")
    ### Plot the predicted image
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    plt.tight_layout()
    im0 = axs[0].imshow(interpolated_approx, vmin=gmin, vmax=gmax)
    axs[0].set_title("Interpolated Image (RBF Functions)")

    im1 = axs[1].imshow(sparse_groundtruth, vmin=gmin, vmax=gmax)
    axs[1].set_title("Sanity Image")

    im1 = axs[2].imshow(predicted_image, vmin=gmin, vmax=gmax)
    axs[2].set_title("ML-Inferred Values")

    # dump_to_log(predicted_image, "predicted_image", "predicted_image.log")
    fig.colorbar(im0, ax=axs, orientation="vertical", label="Thickness", shrink=0.5)

    # for i in range(len(axs)):
    #     axs[i].set_ylim(100, 200)
    #     axs[i].set_xlim(0, 200)
    plt.show()
