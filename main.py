import enum
import os
from argparse import ArgumentParser
from collections import namedtuple
from math import pi, sqrt
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import set_detect_anomaly  # type: ignore
from tqdm import tqdm

from hyptraining.datap.data import preprocess_data
from hyptraining.utils.utils import create_logger, unused, compare_picture


# Create Enum for filetype
class FileType(enum.Enum):
    CSV = "csv"
    PARQUET = "parquet"
    ENVI = "envi"


STR_TO_FILE = {k.value: k for k in FileType}

from image_processing import (
    Point,
    chords_to_circle,
    find_distinctive_feature_coords,
    get_final_image,
)
from model import Model

set_detect_anomaly(True)

DataSet = List[List[Any]]

# For debugging later
logger = create_logger(os.path.abspath(__file__))

set_detect_anomaly(True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
set_detect_anomaly(True)


def arguments():
    """
    Setup Arguments
    """
    # Parameters for data retrieval
    ap = ArgumentParser()
    ap.add_argument("--model_name", default="weights")
    ap.add_argument(
        "--rawdata_path",
        default="./data/raw/",
        help="Directory where objective and hyperspectral data will be stored",
    )
    ap.add_argument(
        "--cache_path",
        default="./data/cache/",
        help="Where to store post-processed data.",
    )
    ap.add_argument(
        "--model_path",
        default="./models",
        help="Where to store trained data",
    )

    ap.add_argument(
        "--image_channels", default=120, type=int, help="Size of model's input"
    )
    ap.add_argument("--image_height", default=1280, help="Hyperspectral image height.")
    ap.add_argument("--image_width", default=1024, help="Hyperspectral image width")
    ap.add_argument(
        "--target_image_size",
        default=150 * 2,
        help="Size of the image containing target information (Used for normalization).",
    )
    ap.add_argument(
        "--feature_angle",
        default=pi / 2,
        help="How we want the feature to be oriented on all data points.",
    )
    ap.add_argument(
        "--template_location",
        default="./feature_right.png",
        help="Template for detecting features",
    )

    # Parmeters for training
    ap.add_argument("--epochs", default=30, type=int, help="Training Epochs")
    ap.add_argument("--batch_size", default=32, help="Batch Size")
    ap.add_argument("--random_seed", default=42, help="Seed for psudo-randomness")
    ap.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight Decay for L2 Regularization",
    )
    ap.add_argument(
        "--train_val_split", default=[0.8, 0.2], help="Train, Test Split"  # Val later
    )

    return ap.parse_args()


@unused
def build_learning_ds(
    ds_thickness: pd.DataFrame,
    ds_hyper: pd.DataFrame,
    hyp_img_height: int,
    hyp_img_width: int,
) -> List[List[Any]]:
    ds = []
    """
    Takes both datasets and tries to create samples in form 
    of List[band0,band1,...,band121, thickness] in order to better suit learning algorithm
    """

    for row_idx, row in ds_thickness.iterrows():
        # x, y = thickness_to_hyper_coords(
        #     row["X"], row["Y"], hyp_img_height, hyp_img_width
        # )
        # Calculate indices for flat list
        ds_hyper_idx = x * hyp_img_width + y
        hyperspec_vec = ds_hyper.iloc[ds_hyper_idx, 2:]
        if x is not None and y is not None:
            # Add to X
            ds.append(hyperspec_vec.tolist() + [row["Thickness"]])
        else:
            logger.warn("Could not find some of the corresponding coordinates")

    return ds


def infill_missing_point(row: pd.Series):
    # TODO: Fill in with the right type of content
    pass


def N_channels_to_one(image: np.ndarray) -> np.ndarray:
    # TODO: You can apply whatever technique you want here to change it from n channels to 1
    # Ill just to mean for now
    return np.mean(image, axis=2)
    # You could also just select one of the channels in question


# def load_cached_data(cache_path: str) -> pd.DataFrame:
def load_cached_data(cache_path: str) -> List[List[Tuple]]:
    # Ensure .cache path exists
    assert os.path.exists(cache_path) and os.path.isdir(
        cache_path
    ), "Cache path does not exist. This error should not happen"

    # Iterate through the cache path
    load_bar = tqdm(
        total=len([f for f in os.listdir(cache_path) if f.endswith(".parquet")]),
        desc="Loading Cache",
    )
    samples: List[List] = []
    for file in os.listdir(cache_path):
        # Check if the file is an .npy
        if ".parquet" in file:
            file_path = os.path.join(cache_path, file)
            df = pd.read_parquet(file_path)
            for _, row in df.iterrows():
                samples.append(row[4:].tolist())
        load_bar.update(1)
    logger.info(f"Obtained a total of {len(samples)} files")

    frame = pd.DataFrame(samples).sample(frac=1).reset_index(drop=True)
    print(f"frame shape is {frame.shape}")
    # Just give me back a list
    simple_list = frame.values.tolist()
    return simple_list


if __name__ == "__main__":
    args = arguments()
    # Set random seeds
    torch.manual_seed(args.random_seed)

    # Parquet format is a lot lighter than csv
    logger.info(f"Preprocesing data at {args.rawdata_path}")

    # Look for new data and convert to data that is easier to process
    preprocess_data(
        args.rawdata_path,
        args.cache_path,
        args.template_location,
        args.image_width,  # TODO: perhaps read this from the files themselves
        args.image_height,
        args.image_channels,
        args.feature_angle,
        args.target_image_size,
    )

    logger.info("Finished Preprocessing")

    # Load all cached files
    dataset = load_cached_data(args.cache_path)
    # Shuffle dataset
    dataset = np.random.permutation(dataset).tolist()

    logger.info("Dataset loaded")
    # Sample Uniformly
    # Create model-related objects
    model = Model(args.image_channels, 1)
    logger.info("Model Structure looks like:")
    logger.info(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=args.weight_decay
    )
    criterium = torch.nn.MSELoss()
    # TODO: scheduler  = ... (if necessary)

    # Build X -> Y Dataset
    # dataset = build_learning_ds(ds_a, ds_b, args.image_height, args.image_width)
    # Create Train-Validation split

    ds_train = dataset[: int(len(dataset) * args.train_val_split[0])]
    ds_val = dataset[int(len(dataset) * args.train_val_split[0]) :]

    # TODO: Add support for missing points using compressed sensing

    # Train Loop
    logger.info(f"Length of train dataset is {len(ds_train)}")
    logger.info(f"Length of validation dataset is {len(ds_val)}")
    num_train_batches = len(ds_train) // args.batch_size  # Throwing remainder
    num_val_batches = len(ds_val) // args.batch_size  # Throwing remainder
    # For graphing later
    train_losses = []
    val_losses = []

    ########################################
    # Main Training Loop
    ########################################

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        batch_losses = []

        ## Training
        model.train()
        for b in range(num_train_batches):
            optimizer.zero_grad()
            batch = torch.Tensor(
                ds_train[b * args.batch_size : (b + 1) * args.batch_size]
            )
            y = batch[:, 0]
            x = batch[:, 1:]

            # Train
            y_pred = model(x).squeeze()
            loss = criterium(y_pred, y).mean()
            loss.backward()
            optimizer.step()
            batch_losses.append(sqrt(loss.item()))

        train_losses.append(sum(batch_losses) / len(batch_losses))

        ## Validation
        model.eval()
        with torch.no_grad():
            val_batch = torch.Tensor(ds_val)
            x = val_batch[:, 1:]
            y = val_batch[:, 0]

            y_pred = model(x).squeeze()
            val_loss = criterium(y_pred, y).mean()
            val_losses.append(sqrt(val_loss.item()))
            # Calculate R^2
            ss_res = torch.sum((y - y_pred) ** 2)
            ss_tot = torch.sum((y - torch.mean(y)) ** 2)
            rsqrd = 1 - ss_res / ss_tot

            logger.info(f"val_loss {sqrt(val_loss.item())}")
            logger.info(f"R^2  {rsqrd.item()}")

    # Show Train-Test Performance
    # logger.info(f"Train Losses: {train_losses}")
    # logger.info(f"Val Losses: {val_losses}")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Losses")
    plt.plot(val_losses, label="Val Losses")
    plt.legend()
    plt.title("RMSE Loss vs Epochs")
    plt.ylabel("RMSE Loss")
    plt.xlabel("Training Epochs")
    plt.show()

    model_path = os.path.join(args.model_path, f"{args.model_name}.pth")
    decision = input(f"Would you like to save this model (to {model_path})? (y/N): ")
    if decision.lower() == "y":
        os.makedirs(args.model_path, exist_ok=True)
        metadata_path = os.path.join(args.model_path, f"{args.model_name}.metadata")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved at {model_path}")
        model_dict_str = str(model)
        with open(metadata_path, "w") as f:
            f.write(model_dict_str)
