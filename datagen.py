"""
Script to generate format we expect.
All data is generated at random
For sources see:
- https://courses.physics.ucsd.edu/2011/Summer/session1/physics1c/lecture10.pdf
"""
import math
import os
from argparse import ArgumentParser
from math import pi
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from hyptraining.utils.utils import Point

# Assume refractive indices for silicon and air
n_sil = 3.5
n_air = 1.0

bands_lims = np.linspace(400e-9, 1000e-9, 122 + 1)

# Air-Film Fresnel Reflectance
affr = (n_air - n_sil) / (n_air + n_sil)
affr = affr**2


def getargs():
    ap = ArgumentParser()
    ap.add_argument("--save_dir", default="./data/raw/")
    ap.add_argument("--resolution", default=[1280, 1024], type=List)
    ap.add_argument("--num_samples", default=885, type=int)
    ap.add_argument("--num_bands", default=122, type=int)
    ap.add_argument("--should_noise", default=True, type=bool)

    # Devil's Details
    ap.add_argument("--aoi_offset", default=0.1, type=float)
    ap.add_argument("--blemish_angle", default=7 * pi / 5, type=float)
    ap.add_argument("--blemish_distance_p_aoiradius", default=0.50, type=float)
    ap.add_argument("--feature_img_path", default="./feature_right.png", type=str)
    ap.add_argument("--aoi_radius_p_res", default=0.85, type=float)
    ap.add_argument("--blemish_radius", default=0.1, type=float)

    return ap.parse_args()


def sample_thick_map(thick_map: np.ndarray, samples=885):
    """
    Will return only 885 samples normalized in coordinates
    """
    set_of_tuples = {}
    while len(set_of_tuples) < samples:
        i = np.random.randint(0, thick_map.shape[0] - 1)
        j = np.random.randint(0, thick_map.shape[1] - 1)
        set_of_tuples[(i, j)] = thick_map[i, j]
    idxs = np.array([list(k) for k in set_of_tuples.keys()], dtype=np.float32)

    vals = np.array(list(set_of_tuples.values()))
    final_vals = np.concatenate((idxs, vals.reshape(-1, 1)), axis=1)

    return final_vals


def rotate_image(img: np.ndarray, radians: float):
    """
    Rotate an image by a given angle
    """
    import cv2

    # Get the center of the image
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, np.degrees(radians), 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def ds_creation(
    res: List[int],
    num_bands: int,
    should_noise: bool,
    aoi_offset: Point,
    blemish_angle: float,
    feature_img: np.ndarray,
    blemish_distance_p_aoiradius: float = 0.45,
    aoi_radius_p_res: float = 0.84,
    blemish_radius_paoiradius: float = 0.10,
):
    """
    Will create dataset with thicknesses initialized (uniformly) at random
    will then use Fresnel equations + Diffraction Equations to establish relationship
    between thickness and resulting wave bands.
    Will return a (Height x Width x # WaveBands(122)) tensor

    Arguments
    ~~~~~~~~~

    Returns
    ~~~~~~~~~
    """
    smalled_dim = min(res)
    aoi_radius = (smalled_dim // 2) * aoi_radius_p_res
    ########################################
    # Thickness Map
    ########################################
    # Make it uniform for now
    thick_map = np.random.uniform(size=(res[0], res[1])) * 1e-3

    # 2D Gaussian centered at image's center
    # thick_map = np.zeros((res[0], res[1]))
    # i, j = np.indices((res[0], res[1]))
    # thick_map = np.exp(
    #     -((i - res[0] // 2) ** 2 + (j - res[1] // 2) ** 2) / (2 * 100**2)
    # )

    # Offset is randomw ithin range
    aoi_offset_max = smalled_dim * aoi_radius_p_res // 2 + feature_img.shape[0] // 2
    aoi_offset = Point(
        np.random.uniform(aoi_offset_max, res[1] - aoi_offset_max),
        np.random.uniform(aoi_offset_max, res[0] - aoi_offset_max),
    )

    # Assert circle within image
    assert (
        aoi_offset.x - aoi_radius > 0 and aoi_offset.y - aoi_radius > 0
    ), "Circle out of bounds"
    assert (
        aoi_offset.x + aoi_radius < res[1] and aoi_offset.y + aoi_radius < res[0]
    ), "Circle out of bounds"

    # Only keep a circle of uniform thickness, the rest gets blacked out
    i, j = np.indices((res[0], res[1]))
    circle = (i - aoi_offset.y) ** 2 + (j - aoi_offset.x) ** 2 < aoi_radius**2
    thick_map[~circle] = 0

    ########################################
    # HyperImage
    ########################################
    hyper_img = np.empty(shape=(res[0], res[1], num_bands))

    # Generate Samples
    bar = tqdm(total=num_bands, desc="Creating power for band: ")
    # for i in range(len(bands_lims) - 1):
    for i in range(num_bands):  # TODO: change this back to 122
        reflec_power = band_calculation(i, thick_map, should_noise)
        hyper_img[:, :, i] = reflec_power
        bar.update(1)

    # Feature Orientation
    random_feature_orientation = np.random.choice([0, pi / 2, pi, 3 / 2 * pi])
    print(f"Random feature orientation is at {np.degrees(random_feature_orientation)}")

    # Place feature at some point in the edge.
    feature_location = Point(
        int(aoi_offset.x + aoi_radius * np.cos(random_feature_orientation)),
        int(aoi_offset.y - aoi_radius * np.sin(random_feature_orientation)),
    )

    # Feature points to 0 degrees, rotate it to random_feature_orientation
    # Apply transformation to array based on random_feature_orientation
    feature_img = rotate_image(feature_img, random_feature_orientation)
    feature_img = np.round(feature_img / 255)

    # Change hyper_img
    feature_insertion_img = hyper_img[
        feature_location.y
        - feature_img.shape[0] // 2 : feature_location.y
        + feature_img.shape[0] // 2,
        feature_location.x
        - feature_img.shape[1] // 2 : feature_location.x
        + feature_img.shape[1] // 2,
        :,
    ]
    # TODO: Clean this mess up
    multi = np.multiply(
        feature_insertion_img,
        np.stack([feature_img] * hyper_img.shape[2], axis=-1),
    )
    amnt = np.sum(feature_img == 0)
    multi[feature_img == 0] = np.stack(
        [band_calculation(1, np.full((amnt,), 0), should_noise)] * hyper_img.shape[2],
        -1,
    )
    hyper_img[
        feature_location.y
        - feature_img.shape[0] // 2 : feature_location.y
        + feature_img.shape[0] // 2,
        feature_location.x
        - feature_img.shape[1] // 2 : feature_location.x
        + feature_img.shape[1] // 2,
        :,
    ] = multi

    # Add a blemish
    # Within the circle add a blemish
    blemish_distance = blemish_distance_p_aoiradius * aoi_radius
    blemish_point = Point(
        aoi_offset.x
        + blemish_distance * np.cos(blemish_angle + random_feature_orientation),
        aoi_offset.y
        - blemish_distance * np.sin(blemish_angle + random_feature_orientation),
    )
    # Add black circle spot at blemish_points
    i, j = np.indices((res[0], res[1]))
    blemish_circle_idx = (i - blemish_point.y) ** 2 + (j - blemish_point.x) ** 2 < (
        blemish_radius_paoiradius * aoi_radius
    ) ** 2
    hyper_img[blemish_circle_idx] = 0

    avg_hyper_image = np.mean(hyper_img, axis=-1)

    # Show Image
    cv2.imshow("HyperImage", hyper_img[:, :, 0])
    cv2.waitKey(0)

    return thick_map, hyper_img


def band_calculation(idx: int, thick_map: np.ndarray, should_noise: bool):
    """
    Will calculate the reflectance of a single band by sampling 15 inner points and averaging
    their power.
    Adding noise to data may be toggled.
    """
    band0 = bands_lims[idx]
    bandf = bands_lims[idx + 1]

    samples = np.linspace(band0, bandf, 15)
    phase_delta = np.expand_dims(
        2 * np.pi * n_sil * thick_map, axis=-1
    ) / np.expand_dims(samples, axis=[0, 1])

    interference_term = 1 + np.cos(phase_delta)
    reflectances = affr * interference_term

    reflectance = np.mean(reflectances, axis=-1)

    # Noise
    noise = np.ones_like(phase_delta)
    if should_noise:
        variance = np.var(reflectances, axis=-1)
        noise = np.random.normal(0, variance * 10, size=reflectances.shape[:-1])
        reflectance += noise

    return reflectance


if __name__ == "__main__":
    args = getargs()

    print("Creating ./data")
    os.makedirs(args.save_dir, exist_ok=True)

    date_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    thickness_path = Path(args.save_dir) / f"target_{date_time}.parquet"
    hyperspec_path = Path(args.save_dir) / f"features_{date_time}.parquet"

    # Load Feature image(Single channel)
    feat_img: np.ndarray = cv2.imread(args.feature_img_path, cv2.IMREAD_GRAYSCALE)

    # Create Datasets
    print("Creating datasets")
    thick_map, hyper_img = ds_creation(
        args.resolution,
        args.num_bands,
        args.should_noise,
        args.aoi_offset,
        args.blemish_angle,
        feat_img,
        args.blemish_distance_p_aoiradius,
        args.aoi_radius_p_res,
        args.blemish_radius,
    )
    # SubSample Dataset
    print("Subsampling Data")
    sampled_thick_map = sample_thick_map(thick_map)
    # sampled_thick_map[:, 0] /= args.resolution[0]
    # sampled_thick_map[:, 1] /= args.resolution[1]

    print("Saving thickness data")
    columns_A = ["X", "Y", "Thickness"]
    pd.DataFrame(sampled_thick_map, columns=columns_A).to_parquet(
        thickness_path, index=False
    )

    # Process data as final presentation format
    i, j = np.indices((1280, 1024))
    ix, jx = (np.expand_dims(i, -1), np.expand_dims(j, -1))
    print(
        f"Hyper image shape {hyper_img.shape} whereas ix, jx are {ix.shape}, {jx.shape}"
    )
    hyper_idxd_and_squeezed = np.concatenate((ix, jx, hyper_img), axis=-1).reshape(
        -1, 2 + args.num_bands
    )

    print("Creating HyperSpectral Data")
    columns_B = ["U", "V"] + [f"hyp{i}" for i in range(122)]
    pd.DataFrame(hyper_idxd_and_squeezed, columns=columns_B).to_parquet(
        hyperspec_path, index=False
    )

    print("Done")
