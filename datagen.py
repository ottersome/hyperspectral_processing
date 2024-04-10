"""
Script to generate format we expect.
All data is generated at random
For sources see:
- https://courses.physics.ucsd.edu/2011/Summer/session1/physics1c/lecture10.pdf
"""
import math
import os
from argparse import ArgumentParser
from math import floor, pi
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hyptraining.utils.distributions import (
    sample_distribution_per_pixel,
    sample_studentt,
    student_t_show,
)
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
    ap.add_argument("--resolution", default=[1024, 1280], type=List[int])  # type:ignore
    ap.add_argument("--num_samples", default=885, type=int)
    ap.add_argument("--num_bands", default=122, type=int)
    ap.add_argument(
        "--should_noise", default=False, type=bool
    )  # Looks too noisy with the extra noise
    ap.add_argument("--data_points", default=5, type=int)
    ap.add_argument("--show_image", action="store_true")

    # Devil's Details
    ap.add_argument("--aoi_offset", default=0.1, type=float)
    ap.add_argument("--blemish_angle", default=7 * pi / 5, type=float)
    ap.add_argument("--blemish_distance_p_aoiradius", default=0.50, type=float)
    ap.add_argument("--feature_img_path", default="./feature_right.png", type=str)
    ap.add_argument("--aoi_radius_p_res", default=0.85, type=float)
    ap.add_argument("--blemish_radius", default=0.1, type=float)
    ap.add_argument(
        "--thickness_variance",
        default=0.04,
        type=float,
        help="Models how much the thickness changes around mean.",
    )
    ap.add_argument(
        "--num_thickness_smaples",
        default=885,
        type=int,
        help="How many samplews to extract from thickness",
    )
    ap.add_argument(
        "--thickimage_radius",
        default=150,
        help="Radius of image centered at 0,0 representing thickness",
    )

    return ap.parse_args()


def sample_thick_map(
    thick_map: np.ndarray,
    aoi_offset: Point,
    aoi_radius: float,
    samples=885,
):
    """
    Will return only 885 samples normalized in coordinates
    """
    set_of_tuples = {}
    while len(set_of_tuples) < samples:
        theta = np.random.uniform(0, 2 * np.pi)
        random_radius = np.random.uniform(0, aoi_radius)

        i = int(aoi_offset.y - random_radius * np.sin(theta))
        j = int(aoi_offset.x + random_radius * np.cos(theta))

        set_of_tuples[(i, j)] = thick_map[i, j]

    idxs = np.array([list(k) for k in set_of_tuples.keys()], dtype=np.float32)

    vals = np.array(list(set_of_tuples.values()))
    final_vals = np.concatenate((idxs, vals.reshape(-1, 1)), axis=1)

    return final_vals


def imagecoords_to_thickcoords(
    points: np.ndarray, img_aoi_center: Point, img_aoi_radius: float, thick_radius=150
) -> np.ndarray:
    """
    Points: nx2 matrix where first column is i and second is j
    """
    # Create scale matrix(img to thick coordinates)
    translate_vector = np.array([-img_aoi_center.y, -img_aoi_center.x])
    pointst = points + np.stack([translate_vector] * points.shape[0], axis=0)
    pointst = np.multiply(pointst, np.stack(([-1, 1],) * points.shape[0], axis=0))
    scale_matrix = np.array(
        [[thick_radius / img_aoi_radius, 0], [0, thick_radius / img_aoi_radius]]
    )
    new_scaled = scale_matrix @ pointst.T

    return new_scaled.T


def rotate_image(img: np.ndarray, radians: float):
    """
    Rotate an image by a given angle
    """
    import cv2

    # Get the center of the image
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, np.degrees(radians), 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def _generate_thickness(
    width: int,
    height: int,
    aoi_offset: Point,
    aoi_radius: float,
    offset: float = 0.5,
    normal_scale=0.04,
) -> np.ndarray:
    width, height = (width, height)

    img = np.zeros((height, width))
    for i in tqdm(range(height)):
        for j in range(width):
            s0 = sample_distribution_per_pixel(
                i,
                j,
                aoi_offset,
                aoi_radius,
                normal_scale,
                offset,
            )
            img[i, j] = s0

    return img


# TODO: Break this function apart, arguments are exploding
def ds_creation(
    resolution: Point,
    num_bands: int,
    should_noise: bool,
    aoi_offset: Point,
    blemish_angle: float,
    feature_img: np.ndarray,
    show_image: bool = False,
    thickness_variance: float = 0.04,
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
    smalled_dim_val = min(resolution)
    smalled_dim_idx = resolution.index(smalled_dim_val)
    aoi_radius = (
        (smalled_dim_val) // 2 - feature_img.shape[smalled_dim_idx] // 2
    ) * aoi_radius_p_res
    ########################################
    # Thickness Map
    ########################################

    # Offset is randomw ithin range
    aoi_offset_max = aoi_radius + feature_img.shape[smalled_dim_idx] // 2
    aoi_offset = Point(
        np.random.uniform(aoi_offset_max, resolution.x - aoi_offset_max),
        np.random.uniform(aoi_offset_max, resolution.y - aoi_offset_max),
    )

    # Assert circle within image
    assert (
        aoi_offset.x - aoi_radius > 0 and aoi_offset.y - aoi_radius > 0
    ), "Circle out of bounds"
    assert (
        aoi_offset.x + aoi_radius < resolution.x
        and aoi_offset.y + aoi_radius < resolution.y
    ), "Circle out of bounds"

    thick_map = _generate_thickness(
        resolution.x,
        resolution.y,
        aoi_offset,
        aoi_radius,
        normal_scale=thickness_variance,
    )

    # We want to scale thickness to bands magnitudes
    bmin, bmax = (bands_lims[0], bands_lims[-1])
    thick_map = thick_map * (bmax - bmin) + bmin
    # thick_map = thick_map * (bmin - 1e-9) + 1e-9

    ########################################
    # HyperImage
    ########################################
    hyper_img = np.empty(shape=(resolution.y, resolution.x, num_bands))

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
    low_thickness = (
        np.clip(np.random.normal(0, scale=thickness_variance, size=amnt), 0, 1)
        * (bmax - bmin)
        + bmin
    )
    calculation = simple_band_calculation(1, low_thickness, should_noise)
    multi[feature_img == 0] = (
        np.stack(
            [calculation] * hyper_img.shape[2],
            -1,
        )
        + 0.1
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
    i, j = np.indices((resolution.y, resolution.x))
    blemish_circle_idx = (i - blemish_point.y) ** 2 + (j - blemish_point.x) ** 2 < (
        blemish_radius_paoiradius * aoi_radius
    ) ** 2
    hyper_img[blemish_circle_idx] = 0

    avg_hyper_image = np.mean(hyper_img, axis=-1)

    # Show Image
    if show_image:
        # Show where center of aoi is
        # Show Image
        plt.imshow(avg_hyper_image)
        plt.clim(0, 1)
        # Right side value legend
        plt.colorbar()
        plt.show()

    return thick_map, hyper_img, aoi_offset, aoi_radius


def band_calculation(idx: int, thick_map: np.ndarray, should_noise: bool):
    """
    Will calculate the reflectance of a single band by sampling 15 inner points and averaging
    their power.
    Adding noise to data may be toggled.
    """
    band0 = bands_lims[idx]
    bandf = bands_lims[idx + 1]

    samples = np.linspace(band0, bandf, 15)
    twod_samples = np.expand_dims(samples, axis=[0, 1])
    phase_delta = np.expand_dims(2 * np.pi * n_sil * thick_map, axis=-1) / twod_samples

    interference_term = 1 + np.cos(phase_delta)
    reflectances = affr * interference_term
    reflectance = np.mean(reflectances, axis=-1)

    # Normalize reflectace
    reflectance = (reflectance - np.min(reflectance)) / (
        np.max(reflectance) - np.min(reflectance)
    )
    # Noise
    noise = np.ones_like(phase_delta)
    if should_noise:
        variance = np.var(reflectances, axis=-1)
        noise = np.random.normal(0, variance * 10, size=reflectances.shape[:-1])
        reflectance += noise

    return reflectance


def simple_band_calculation(idx: int, thick_map: np.ndarray, should_noise: bool):
    """
    Will calculate the reflectance of a single band by sampling 15 inner points and averaging
    their power.
    Adding noise to data may be toggled.
    """
    band0 = bands_lims[idx]
    bandf = bands_lims[idx + 1]

    samples = np.linspace(band0, bandf, 15)
    twod_samples = np.expand_dims(samples, axis=0)
    phase_delta = np.expand_dims(2 * np.pi * n_sil * thick_map, axis=-1) / twod_samples

    interference_term = 1 + np.cos(phase_delta)
    reflectances = affr * interference_term
    reflectance = np.mean(reflectances, axis=-1)

    # Normalize reflectace
    reflectance = (reflectance - np.min(reflectance)) / (
        np.max(reflectance) - np.min(reflectance)
    )
    # Noise
    noise = np.ones_like(phase_delta)
    if should_noise:
        variance = np.var(reflectances, axis=-1)
        noise = np.random.normal(0, variance * 10, size=reflectances.shape[:-1])
        reflectance += noise

    return reflectance


if __name__ == "__main__":
    args = getargs()

    meep = [2.5, 2.5]
    mean = np.array(meep)  # The location vector (mean)

    print("Creating ./data")
    os.makedirs(args.save_dir, exist_ok=True)

    bar = tqdm(total=args.data_points, desc="Creating datapoints", leave=False)

    for i in range(args.data_points):
        date_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        thickness_path = Path(args.save_dir) / f"target_{date_time}.parquet"
        thickness_subsampled_path = (
            Path(args.save_dir) / f"target_subsampled{date_time}.parquet"
        )
        hyperspec_path = Path(args.save_dir) / f"features_{date_time}.parquet"

        # Load Feature image(Single channel)
        feat_img: np.ndarray = cv2.imread(args.feature_img_path, cv2.IMREAD_GRAYSCALE)

        # Create Datasets
        bar.set_description("Creating datasets")
        thick_map, hyper_img, aoi_offset, aoi_radius = ds_creation(
            Point(args.resolution[1], args.resolution[0]),
            args.num_bands,
            args.should_noise,
            args.aoi_offset,
            args.blemish_angle,
            feat_img,
            args.show_image,
            args.thickness_variance,
            args.blemish_distance_p_aoiradius,
            args.aoi_radius_p_res,
            args.blemish_radius,
        )
        # SubSample Dataset
        bar.set_description(f"Subsampling {args.num_samples} Data from thickness image")
        sampled_thick_map = sample_thick_map(
            thick_map, aoi_offset, aoi_radius, args.num_samples
        )
        # Noramalize to have origin at 0,0 and radius of 150
        ij = sampled_thick_map[:, [0, 1]]
        xy = imagecoords_to_thickcoords(ij, aoi_offset, aoi_radius)

        for x, y in zip(xy[:, 0], xy[:, 1]):
            length = floor(np.sqrt(x**2 + y**2))
            assert (
                length <= args.thickimage_radius
            ), f"過了 with values x:{x},y:{y} and length:{length}"
        sampled_thick_map[:, [0, 1]] = xy

        # sampled_thick_map[:, 0] /= args.resolution[0]
        # sampled_thick_map[:, 1] /= args.resolution[1]

        bar.set_description("Saving thickness data")
        columns_A = ["X", "Y", "Thickness"]
        pd.DataFrame(sampled_thick_map, columns=columns_A).to_parquet(
            thickness_subsampled_path, index=False
        )

        # Process data as final presentation format
        i, j = np.indices((args.resolution[0], args.resolution[1]))
        ix, jx = (np.expand_dims(i, -1), np.expand_dims(j, -1))
        bar.set_description(
            f"Hyper image shape {hyper_img.shape} whereas ix, jx are {ix.shape}, {jx.shape}"
        )
        hyper_idxd_and_squeezed = np.concatenate((ix, jx, hyper_img), axis=-1).reshape(
            -1, 2 + args.num_bands
        )

        bar.set_description("Creating HyperSpectral Data")
        columns_B = ["U", "V"] + [f"hyp{i}" for i in range(args.num_bands)]
        pd.DataFrame(hyper_idxd_and_squeezed, columns=columns_B).to_parquet(
            hyperspec_path, index=False
        )

        bar.update(1)

    print("Done"), args.resolution[1]
