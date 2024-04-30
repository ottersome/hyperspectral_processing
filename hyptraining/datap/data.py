import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from ..utils.utils import Point, create_logger
from .processing import Circle, get_standard_source

file_path = os.path.basename(__file__)
DATA_LOGGER = create_logger(file_path)


def thickness_to_hyper_coords(
    x: Union[np.ndarray, int],
    y: Union[np.ndarray, np.ndarray],  # , img_height, img_width
) -> Point:
    """
    function you may fill with a mapping of:
        (u,v) coordinates in wafer space to (x,y) coordinates in pixel space
    """
    return Point(int(x * (483 / 150) + 515), int(y * (483 / 150) + 518))


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


def combine_srctarg_into_sample(
    src_img: np.ndarray, target: pd.DataFrame, ignore_spot: Circle
) -> List[tuple]:
    """
    Will look at target and find corresponding elements in source to keep
    """
    new_rows = []
    for _, row in target.iterrows():
        x, y = row[["X", "Y"]]
        t = row["SiOTHK___"]
        hyper_point = thickness_to_hyper_coords(x, y)
        # Convert image into array
        if in_circle(hyper_point, ignore_spot):
            print("Found target on blindspot")
            continue  # Just dont use it

        # Source
        ij_features = src_img[hyper_point.x, hyper_point.y, :]

        print(f"ADding {ij_features.shape} channels.")

        debugging_feetures = [
            x,
            y,
            hyper_point.x,
            hyper_point.y,
        ]  # Not really necessary
        new_tuple = debugging_feetures + [t] + ij_features.tolist()
        print(f"ADding tuple of length {len(new_tuple)}")
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
):
    """
    Will check if there is any data that I have to work out:
    """

    # If source_dir exists
    assert (
        Path(rawdata_dir).exists() and Path(rawdata_dir).exists()
    ), "Raw data dir does not exists"

    # Replace the extension with npy (could be any in a range of extensions)
    columns = ["X", "Y", "I", "J", "Thickness"] + [
        f"C{i}" for i in range(source_channels)
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
            source_image = _read_source(os.path.join(rawdata_dir, featurefile_name))

            # Get the final image in the size that we want
            standard_source, ignore_spot = get_standard_source(
                src=source_image,
                template_loc=template_loc,
                src_width=source_width,
                src_height=source_height,
                src_channels=source_channels,
                feature_angle=feature_angle,
            )

            # Form the rows from the target
            print(f"Standard source shape {standard_source.shape}")
            print(f"Targe_rowsshape {target_rows.shape}")
            final_rows = combine_srctarg_into_sample(
                standard_source, target_rows, ignore_spot
            )

            pd.DataFrame(final_rows, columns=columns).to_parquet(
                saveto_path, index=False
            )

            # Save to final_rows to some file
            DATA_LOGGER.info(f"`{file}` added to cache  as `{saveto_path}`")

    DATA_LOGGER.info("Finished preprocessing")


##################################################
# VERY implementation heavy after this point
##################################################


def _read_source(file_path: str) -> pd.DataFrame:
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


def read_csv(csv_path: str, file_t):
    # Will assume a fixed data format
    if "source" in csv_path:
        pass
    elif "target" in csv_path:
        pass
    else:
        raise ValueError("Unknown file type. Neither hyperspec, nor thickness")
