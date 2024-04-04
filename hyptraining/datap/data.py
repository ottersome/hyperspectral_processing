import os
from typing import List

import numpy as np
import pandas as pd

from ..utils.utils import create_logger
from .processing import get_standard_source

file_path = os.path.basename(__file__)
DATA_LOGGER = create_logger(file_path)


def thickness_to_hyper_coords(x, y, img_height, img_width) -> tuple[int, int]:
    """
    function you may fill with a mapping of:
        (u,v) coordinates in wafer space to (x,y) coordinates in pixel space
    """
    return (x * (483 / 150) + 515, y * (483 / 150) + 518)


def combine_srctarg_into_sample(
    src_img: np.ndarray, target: pd.DataFrame
) -> List[tuple]:
    """
    Will look at target and find corresponding elements in source to keep
    """
    new_rows = []
    for _, row in target.iterrows():
        x, y = row[["X", "Y"]]
        t = row["Thickness"]
        i, j = thickness_to_hyper_coords(x, y, src_img.shape[1], src_img.shape[0])

        ij_features = src_img[i, j, :]

        debugging_feetures = [x, y, i, j]  # Not really necessary
        new_tuple = debugging_feetures + [t] + ij_features.tolist()
        new_rows.append(new_tuple)
        # TODO: maybe add more stuff to the row

    return new_rows


def preprocess_data(
    source_dir: str,
    cached_dir: str,
    template_img: np.ndarray,
    source_width: int,
    source_height: int,
    source_channels: int,
    feature_angle: float,
):
    """
    Will check if there is any data that I have to work out:
    """

    # If source_dir exists
    assert os.path.exists(source_dir) and os.path.isdir(
        source_dir
    ), "Source dir does not exists"

    # Replace the extensionw with npy (could be any in a range of extensions)
    columns = ["X", "Y", "I", "J"] + [f"C{i}" for i in range(source_channels)]

    # Ensure it exists
    os.makedirs(cached_dir, exist_ok=True)

    # Iterate through source_dir:
    for file in os.listdir(source_dir):
        # Check if the file is a csv
        if "target" in file:
            # Check if cached_dir does not have the equivalent
            cache_name = os.path.basename(file).replace("target", "source")
            cache_name = cache_name[: cache_name.find(".")]
            saveto_path = cached_dir[: cached_dir.find(".")] + ".npy"
            if os.path.exists(os.path.join(cached_dir, file)):
                DATA_LOGGER.info(f"{file} is already cached in {saveto_path}")
                continue
            DATA_LOGGER.info(f"Found unprocessed file {file}. Loading into cache...")

            target_rows = _read_target(os.path.join(source_dir, file))

            # Find Corresponding src file
            source_name = file.replace("target", "features")

            source_image = _read_source(os.path.join(source_dir, source_name))

            # Get the final image in the size that we want
            standard_source: np.ndarray = get_standard_source(
                src=source_image,
                template_img=template_img,
                src_width=source_width,
                src_height=source_height,
                src_channels=source_channels,
                feature_angle=feature_angle,
            )

            # Form the rows from the target
            final_rows = combine_srctarg_into_sample(standard_source, target_rows)

            pd.DataFrame(final_rows).to_parquet(saveto_path, columns=columns)

            # Save to final_rows to some file
            DATA_LOGGER.info(f"`{file}` added to cache  as `{saveto_path}`")

    # ds_a = pd.read_parquet(args.thick_loc)  # Dataset A

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
