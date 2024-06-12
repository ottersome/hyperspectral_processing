import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..utils.utils import Point, create_logger
from .processing import Circle, get_standard_source, thickness_to_hyper_coords

file_path = os.path.basename(__file__)
DATA_LOGGER = create_logger(file_path)


def in_circle(point: Point, circle: Circle) -> bool:
    """
    Check if a point is within a circle
    """
    return (point.x - circle.center.x) ** 2 + (
        point.y - circle.center.y
    ) ** 2 <= circle.radius**2


def nchannel_img_to_array(
    img: np.ndarray,
) -> pd.DataFrame:
    # Flatten it out so that channels are columns in second axis and add
    # new i,j columns that index it
    # img is 3d
    i, j = np.indices(img.shape[:2])
    img_reshape = img.reshape(-1, img.shape[2])
    tabular_img = np.dstack([i, j, img_reshape])
    cols = ["i", "j"] + [f"c{i}" for i in range(img.shape[2])]
    dataframe = pd.DataFrame(tabular_img, columns=cols)  # type:ignore
    return dataframe


def get_roi_around_point(
    point_of_interest: Point,
    kernel_radius: int,
    image: np.ndarray,
    ignore_spot: Circle,
):
    assert len(image.shape) >= 2, "Image needs to be of at least 2d"
    # Get the the extremes
    poi = point_of_interest

    y_rawmin = poi.y - kernel_radius
    y_rawmax = poi.y + kernel_radius
    x_rawmin = poi.x - kernel_radius
    x_rawmax = poi.x + kernel_radius

    ymin = max(y_rawmin, 0)
    ymax = min(y_rawmax, image.shape[0] - 1)
    xmin = max(x_rawmin, 0)
    xmax = min(x_rawmax, image.shape[1] - 1)

    offset_y_left = ymin - y_rawmin
    # offset_y_right = offset_y_left + (ymax - ymin + 1)
    offset_x_left = xmin - x_rawmin
    # offset_x_right = offset_x_left + (xmax - xmin + 1)
    offset_x_right = xmax - x_rawmax
    offset_y_right = ymax - y_rawmax
    offset_x_right = None if offset_x_right == 0 else offset_x_right
    offset_y_right = None if offset_y_right == 0 else offset_y_right

    # Ensure no blind spots are included
    yroi, xroi = np.meshgrid(
        np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1), indexing="ij"
    )
    distances = (yroi - ignore_spot.center.y) ** 2 + (xroi - ignore_spot.center.x) ** 2
    mask = distances > ignore_spot.radius**2
    fmask = mask.astype(float)
    fmask[~mask] = np.nan
    # OPTIM: Check if we can broadcast this
    fmask = np.repeat(fmask[:, :, np.newaxis], image.shape[2], axis=2)

    # Do the actual selection
    return_kernel = np.zeros(
        (kernel_radius * 2 + 1, kernel_radius * 2 + 1, image.shape[2])
    )
    unmasked_selection = image[ymin : ymax + 1, xmin : xmax + 1, :]
    return_kernel[offset_y_left:offset_y_right, offset_x_left:offset_x_right, :] = (
        unmasked_selection * fmask
    )

    return return_kernel


def combine_srctarg_into_sample(
    src_img: np.ndarray,
    target: pd.DataFrame,
    ignore_spot: Circle,
    tgtimg_size: int,
    kernel_radius: int,
) -> List[tuple]:
    """
    Will look at target and find corresponding feature elements to pair it with
    """
    assert (
        src_img.shape[0] == src_img.shape[1]
    ), f"Expecting square image, instead we got {src_img.shape}"

    hypimg_size = src_img.shape[0]

    new_rows = []
    for _, row in target.iterrows():
        x, y = row[["X", "Y"]]
        t = row["SiOTHK___"]
        hyper_point = thickness_to_hyper_coords(x, y, hypimg_size, tgtimg_size)
        hyper_kernel = get_roi_around_point(
            hyper_point, kernel_radius, src_img, ignore_spot
        )

        # Source
        # ij_features = src_img[hyper_point.x, hyper_point.y, :]
        ij_features = hyper_kernel.ravel()

        debugging_feetures = [
            x,
            y,
            hyper_point.x,
            hyper_point.y,
        ]  # Not really necessary
        new_tuple = debugging_feetures + [t] + ij_features.tolist()
        new_rows.append(new_tuple)
        # TODO: maybe add more stuff to the row

    return new_rows


def preprocess_data(
    rawdata_dir: str,
    cached_dir: str,
    template_loc: str,
    source_width: int,
    source_height: int,
    source_channels: int,
    feature_angle: float,
    trgimg_size: int,
    kernel_radius: int,
):
    """
    Will check if there is any data that I have to work out:
    """

    # If source_dir exists
    assert (
        Path(rawdata_dir).exists() and Path(rawdata_dir).exists()
    ), "Raw data dir does not exists"

    # Replace the extension with npy (could be any in a range of extensions)
    num_columns = ((kernel_radius * 2 + 1) ** 2) * source_channels
    columns: List = ["X", "Y", "I", "J", "Thickness"] + [
        f"C{i}" for i in range(num_columns)
    ]

    # Ensure it exists
    Path(cached_dir).mkdir(parents=True, exist_ok=True)

    # Iterate through source_dir:
    for file in os.listdir(rawdata_dir):
        #  start with target file, find the coressopnding feature file.
        if "target" in file:
            # Check if its already been processed
            print(f"Cached dir is {cached_dir}")
            wrong_path = str(Path(cached_dir) / file)
            saveto_path = wrong_path[: wrong_path.find(".")] + ".parquet"

            if os.path.exists(saveto_path):
                DATA_LOGGER.info(f"{file} is already cached in {saveto_path}")
                continue

            # If not already preprocessed:
            DATA_LOGGER.info(f"Found unprocessed file {file}. Loading into cache...")

            # Read target data
            target_rows = _read_target(os.path.join(rawdata_dir, file))

            # Find corresponding feature file
            featurefile_name = file.replace("target", "features")
            source_image = read_source(os.path.join(rawdata_dir, featurefile_name))

            # Get the final image in the size that we want
            standard_source, ignore_spot = get_standard_source(
                src=source_image,
                template_loc=template_loc,
                src_width=source_width,
                src_height=source_height,
                feature_angle=feature_angle,
            )

            # Form the rows from the target
            print(f"Standard source shape {standard_source.shape}")
            print(f"Targe_rowsshape {target_rows.shape}")
            final_rows = combine_srctarg_into_sample(
                standard_source, target_rows, ignore_spot, trgimg_size, kernel_radius
            )

            pd.DataFrame(final_rows, columns=columns).to_parquet(  # type:ignore
                saveto_path, index=False
            )

            # Save to final_rows to some file
            DATA_LOGGER.info(f"`{file}` added to cache  as `{saveto_path}`")

    DATA_LOGGER.info("Finished preprocessing")


##################################################
# VERY implementation heavy after this point
##################################################


def read_source(file_path: str) -> pd.DataFrame:
    """
    Ensure that regardless of source file we get the same type of dataframe out of it
    """
    frame = pd.DataFrame()
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".ENVI"):
        pass
        # TODO: need to understand more how envi files work and their format
    return frame


def _read_target(file_path: str) -> pd.DataFrame:
    """
    Ensure that regardless of source file we get the same type of dataframe out of it
    """
    frame = pd.DataFrame()
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".ENVI"):
        pass
        # TODO: need to understand more how envi files work and their format
    return frame


def read_csv(csv_path: str, _):
    # Will assume a fixed data format
    if "source" in csv_path:
        pass
    elif "target" in csv_path:
        pass
    else:

        raise ValueError("Unknown file type. Neither hyperspec, nor thickness")
