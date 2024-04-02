"""
Script to generate format we expect.
All data is generated at random
For sources see:
- https://courses.physics.ucsd.edu/2011/Summer/session1/physics1c/lecture10.pdf
"""
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Assume refractive indices for silicon and air
n_sil = 3.5
n_air = 1.0

bands_lims = np.linspace(400e-9, 1000e-9, 122 + 1)

# Air-Film Fresnel Reflectance
affr = (n_air - n_sil) / (n_air + n_sil)
affr = affr**2


def getargs():
    ap = ArgumentParser()
    ap.add_argument("--save_dir", default="./data")
    ap.add_argument("--resolution", default=[1280, 1024])
    ap.add_argument("--num_samples", default=885)
    ap.add_argument("--num_bands", default=122)
    ap.add_argument("--should_noise", default=True, type=bool)

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


def ds_creation(res: List[int], num_bands: int, should_noise: bool):
    """
    Will create dataset with thicknesses initialized (uniformly) at random
    will then use Fresnel equations + Diffraction Equations to establish relationship
    between thickness and resulting wave bands.
    Will return a (Height x Width x # WaveBands(122)) tensor
    """
    # Make it uniform for now
    thick_map = np.random.uniform(size=(res[0], res[1])) * 1e-3

    # HyperImage
    hyper_img = np.empty(shape=(res[0], res[1], num_bands))

    # Generate Samples
    wl_samples = np.linspace(bands_lims[0], bands_lims[-1], 15 * len(bands_lims))
    bar = tqdm(total=num_bands, desc="Creating power for band: ")
    # for i in range(len(bands_lims) - 1):
    for i in range(num_bands):  # TODO: change this back to 122
        reflec_power = band_calculation(i, thick_map, should_noise)
        hyper_img[:, :, i] = reflec_power
        bar.update(1)

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

    thickness_path = Path(args.save_dir) / "thickness.csv"
    hyperspec_path = Path(args.save_dir) / "hyperspec.csv"

    # Create Datasets
    print("Creating datasets")
    thick_map, hyper_img = ds_creation(
        args.resolution, args.num_bands, args.should_noise
    )

    # SubSample Dataset
    print("Subsampling Data")
    sampled_thick_map = sample_thick_map(thick_map)
    sampled_thick_map[:, 0] /= args.resolution[0]
    sampled_thick_map[:, 1] /= args.resolution[1]

    print("Saving thickness data")
    columns_A = ["X", "Y", "Thickness"]
    pd.DataFrame(sampled_thick_map, columns=columns_A).to_csv(
        thickness_path, index=False
    )

    # Process data as final presentation format
    i, j = np.indices((1280, 1024))
    ix, jx = (np.expand_dims(i, -1), np.expand_dims(j, -1))
    hyper_idxd_and_squeezed = np.concatenate((ix, jx, hyper_img), axis=-1).reshape(
        -1, 2 + args.num_bands
    )

    print("Creating HyperSpectral Data")
    columns_B = ["U", "V"] + [f"hyp{i}" for i in range(122)]
    pd.DataFrame(hyper_idxd_and_squeezed, columns=columns_B).to_csv(
        hyperspec_path, index=False
    )

    print("Done")
