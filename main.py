import logging
import os
from argparse import ArgumentParser
from collections import namedtuple
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.autograd import set_detect_anomaly
from tqdm import tqdm

from model import Model

set_detect_anomaly(True)

DataSet = List[List[Any]]


def create_logger():
    logger = logging.getLogger("MAIN")
    logger.setLevel(logging.DEBUG)
    cwd = os.getcwd()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh = logging.FileHandler(os.path.join(cwd, "./logs"), "w")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def arguments():
    """
    Setup Arguments
    """
    ap = ArgumentParser()
    ap.add_argument(
        "--thick_loc",
        default="./data/thickness.parquet",
        help="Dataset A (Hyperspectra) location (Assuming CSV)",
    )
    ap.add_argument(
        "--hyp_loc",
        default="./data/hyperspec.parquet",
        help="Dataset B Location (Assuming CSV)",
    )
    ap.add_argument("--input_size", default=122, help="Size of model's input")
    ap.add_argument("--image_height", default=1280, help="Hyperspectral image height.")
    ap.add_argument("--image_width", default=1024, help="Hyperspectral image width")
    ap.add_argument("--epochs", default=10, type=int, help="Training Epochs")
    ap.add_argument("--batch_size", default=32, help="Batch Size")
    ap.add_argument("--random_seed", default=42, help="Seed for psudo-randomness")
    ap.add_argument(
        "--train_val_split", default=[0.8, 0.2], help="Train, Test Split"  # Val later
    )

    return ap.parse_args()


def thickness_to_hyper_coords(
    x, y, img_height, img_width
) -> Union[tuple[int, int], None]:
    """
    function you may fill with a mapping of:
        (u,v) coordinates in wafer space to (x,y) coordinates in pixel space
    """
    return (x * (483 / 150) + 515, y * (483 / 150) + 518)


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
        x, y = thickness_to_hyper_coords(
            row["X"], row["Y"], hyp_img_height, hyp_img_width
        )
        # Calculate indices for flat list
        ds_hyper_idx = x * hyp_img_width + y
        hyperspec_vec = ds_hyper.iloc[ds_hyper_idx, 2:]
        if x is not None and y is not None:
            # Add to X
            ds.append(hyperspec_vec.tolist() + [row["Thickness"]])
        else:
            logger.warn("Could not find some of the corresponding coordinates")

    return ds


def infill_missing_point(row: List[Any]):
    # TODO: Fill in with the right typ of content
    pass

def process_missing_points(ds: DataSet) -> DataSet:
    # This might require something like compressed sensing, matrix reduction,etc
    for row_idx, row in df.iterrows():
        # If we have a missing point, we need to fill it in
        if row.isnull().any():
            pass
        else:
            infill_missing_point(row)
    return ds # CHECK: Ensure this is correct


def find_orientating_feature(template)

if __name__ == "__main__":
    args = arguments()
    # Set random seeds
    torch.manual_seed(args.random_seed)

    logger = create_logger()  # For storing information for debugging

    # Parquet format is a lot lighter than csv
    logger.info("Loading Thickness Dataset A ")
    ds_a = pd.read_parquet(args.thick_loc)  # Dataset A
    ds_a = ds_a.sample(frac=1).reset_index(drop=True)  # Shuffle (best for learning)

    logger.info("Loading Hypersectral Dataset B")
    ds_b = pd.read_parquet(args.hyp_loc)  # Dataset B

    # Create model-related objects
    model = Model(args.input_size, 1)
    logger.info("Model Structure looks like:")
    logger.info(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterium = torch.nn.MSELoss()
    # scheduler  = ... (if necessary)

    # Build X -> Y Dataset
    dataset = build_learning_ds(ds_a, ds_b, args.image_height, args.image_width)
    # Create Train-Validation split
    ds_train = dataset[: int(len(dataset) * args.train_val_split[0])]
    ds_val = dataset[int(len(dataset) * args.train_val_split[0]) :]

    #TODO: Add support for missing points using compressed sensing
    ds_train = process_missing_points(ds_train)
    ds_val = process_missing_points(ds_val)


    # Train Loop
    num_batches = len(ds_train) // args.batch_size  # Throwing remainder
    model.train()
    # For graphing later
    train_losses = []
    val_losses = []

    ########################################
    # Main Training Loop
    ########################################
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        batch_losses = []
        for b in range(num_batches):
            batch = torch.Tensor(
                ds_train[b * args.batch_size : (b + 1) * args.batch_size]
            )
            x = batch[:, :-1]
            y = batch[:, -1]

            # Train
            y_pred = model(x)
            loss = criterium(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_losses.append(sum(batch_losses) / len(batch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            val_batch = torch.Tensor(ds_val)
            x = val_batch[:, :-1]
            y = val_batch[:, -1]
            y_pred = model(x)
            val_loss = criterium(y_pred.squeeze(), y)
            val_losses.append(val_loss.item())
        model.train()


    # Show Train-Test Performance
    # logger.info(f"Train Losses: {train_losses}")
    # logger.info(f"Val Losses: {val_losses}")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Losses")
    plt.plot(val_losses, label="Val Losses")
    plt.legend()
    plt.title("MSE Loss vs Epochs")
    plt.ylabel("MSE Loss")
    plt.xlabel("Training Epochs")
    plt.show()
