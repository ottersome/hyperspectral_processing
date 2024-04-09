import os
from collections import namedtuple
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from ..utils.utils import create_logger

Point = namedtuple("Point", ["x", "y"])
ORIENTATIONS_RAD = [np.pi / 2, np.pi, 3 * np.pi / 2, 0]

logger = create_logger(os.path.basename(__file__))


def find_circle(b, c, d) -> Tuple[Point, int]:
    """
    Returns
    ~~~~~~~
        - circle_center (Point)
        - radius (float)
    """
    B = np.array(
        [
            [(c[0] ** 2 + c[1] ** 2 - b[0] ** 2 - b[1] ** 2) / 2],
            [(d[0] ** 2 + d[1] ** 2 - c[0] ** 2 - c[1] ** 2) / 2],
        ]
    )

    A = np.array([[c[0] - b[0], c[1] - b[1]], [d[0] - c[0], d[1] - c[1]]])
    circle_center_np = np.linalg.inv(A).dot(B).squeeze()
    circle_center = Point(int(circle_center_np[0]), int(circle_center_np[1]))
    print(f"Circle center is {circle_center}")
    return (
        circle_center,
        int(
            np.sqrt((circle_center[0] - b[0]) ** 2 + (circle_center[1] - b[1]) ** 2),
        ),
    )


def chords_to_circle(img: np.ndarray) -> Tuple[Point, int]:
    # Start interactive slection inside of image

    clicks = []

    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
        # At 3 we close the window
        # if len(clicks) >= 3:
        # print("Got all three clicks. Continuing")

    cv2.namedWindow("image")
    # Set Size of window
    cv2.resizeWindow("image", 300, 300)
    cv2.setMouseCallback("image", on_mouse)
    print(f"Image {img}")
    cv2.imshow("image", img)
    # Quit only after three clicks
    while len(clicks) < 3:
        cv2.waitKey(1)

    # Once the windows are distroyed
    coords, circle_radius = find_circle(*clicks)

    return coords, circle_radius


# TODO: find a distinctive feature to base rotation from
def find_distinctive_feature_coords(
    img_gray: np.ndarray, feature_gray: np.ndarray
) -> Point:
    assert len(img_gray.shape) == 2, f"Input image must be 2d"
    assert len(feature_gray.shape) == 2, "Input feature map must be 2D"
    img_gray = img_gray.astype(np.float32)
    feature_gray = feature_gray.astype(np.float32)
    # Find distinctive feature
    # For now we will use the center of the circle
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # feat_gray = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
    print(f"Image size is {img_gray.shape[0]} ,{img_gray.shape[1]}")
    print(f"Feature size is {feature_gray.shape[0]} ,{feature_gray.shape[1]}")
    feat_width, feat_height = feature_gray.shape[0], feature_gray.shape[1]

    cur_maxval = 0
    cur_loc = [-1, -1]

    for r in ORIENTATIONS_RAD:
        score_matrix = cv2.matchTemplate(img_gray, feature_gray, 3)
        rotation = cv2.getRotationMatrix2D(
            (feat_width // 2, feat_height // 2), np.degrees(r), 1
        )
        feature_gray = cv2.warpAffine(feature_gray, rotation, (feat_width, feat_height))
        _, max_val, _, topleft_loc = cv2.minMaxLoc(score_matrix)
        if max_val > cur_maxval:
            cur_maxval = max_val
            cur_loc = topleft_loc

    assert cur_loc[0] != -1, cur_loc[1] != -1

    feat_center = Point(cur_loc[0] + feat_width // 2, cur_loc[1] + feat_height // 2)

    return feat_center


def prompt_for_surrounding_points(image_window_id: str) -> List[Point]:
    """
    WARNING: This will change the image you pass. Please send copy if necessary.
    """
    clicks = []

    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.setMouseCallback(image_window_id, on_mouse)
    # Quit only after three clicks
    while len(clicks) < 3:
        cv2.waitKey(1)
    return clicks


def df_to_img(df: pd.DataFrame, width: int, height: int, channels: int) -> np.ndarray:
    """
    Columns are channels, rows are pixels in row-major
    """
    return df.to_numpy().reshape((height, width, channels))


def get_croppedRotated_img(
    img: np.ndarray,
    feature_location: Point,
    circle_location: Point,
    circle_radius: int,
    preferred_angle: float,
) -> np.ndarray:
    """
    Parameters
    ----------
        - img (np.ndarray): image containing src information,
        - feature_location (Point): where feature of interest is location
        - circle_location (Point): circle locations
        - circle_radius (int): radius of circle
        - preferred_angle (float): angle at which we want our feature
    Returns
    -------
        - final_image: Already cropped image
    """

    # First, crop the image
    cropped = img[
        circle_location.y - circle_radius : circle_location.y + circle_radius,
        circle_location.x - circle_radius : circle_location.x + circle_radius,
    ]
    cropped_height = cropped.shape[0]
    cropped_width = cropped.shape[1]
    # Shift the feature_location
    feature_local = Point(
        feature_location.x - (circle_location.x - cropped_width // 2),
        feature_location.y - (circle_location.y - cropped_height // 2),
    )
    # Ensure feature within radius
    assert (
        feature_local.x < cropped_width and feature_local.y < cropped_height
    ), "Feature not within circle"
    # Angle between args.direction and
    feat_from_center = Point(
        feature_local.x - cropped_width // 2,
        -(feature_local.y - cropped_height // 2),
    )
    print(f"Feature from center {feat_from_center}")
    feature_angle = np.arctan2(feat_from_center.y, feat_from_center.x)
    print(f"Feature angle is rad: {feature_angle} deg: {np.degrees(feature_angle)}")
    print(
        f"Preferred angle is rad: {preferred_angle} deg: {np.degrees(preferred_angle)}"
    )
    rotation_angle = preferred_angle - feature_angle

    rotation_matrix = cv2.getRotationMatrix2D(
        (cropped_height // 2, cropped_width // 2), np.degrees(rotation_angle), 1
    )
    final_image = cv2.warpAffine(
        cropped, rotation_matrix, (cropped_height, cropped_width)
    )

    return final_image


def get_standard_source(
    src: pd.DataFrame,
    template_loc: str,
    src_width: int,
    src_height: int,
    src_channels: int,
    feature_angle: float,  # Should be radians
) -> np.ndarray:
    # Template Image:
    template_img = cv2.imread(template_loc, cv2.IMREAD_GRAYSCALE)
    print(f"Template image size is {template_img.shape}")
    # Select 2-N columns from dataset
    img_columns = src.iloc[:, 2:]
    img: np.ndarray = df_to_img(img_columns, src_width, src_height, src_channels)

    # Get Cropped-in Image
    satisfied = False
    cropped_image = np.ndarray([])

    while not satisfied:
        # Prompt for points of interest
        # gray_img = np.mean(img, axis=-1) if img.shape[2] > 1 else img  # READ ONLY
        gray_img = img[:, :, 0].squeeze() if img.shape[2] > 1 else img  # READ ONLY
        visual_rep = np.stack((gray_img,) * 3, axis=-1).astype(
            np.float32
        )  # Visual changes happen here
        print(f"Show shape {visual_rep.shape}")

        # Show Image for changes
        cv2.namedWindow("image")
        cv2.resizeWindow("image", 300, 300)
        cv2.imshow("image", visual_rep)

        # Ask user to select within the imagek
        print("Please select three points in circle to find radius and center...")
        points = prompt_for_surrounding_points("image")

        # Find Area of Interest
        (
            circle_of_interest_coords,
            circle_of_interest_radius,
        ) = find_circle(*points)

        # Find Feature of Interest
        print(
            f"Center of image found at {circle_of_interest_coords} with radius {circle_of_interest_radius}"
        )

        cv2.circle(
            visual_rep,
            circle_of_interest_coords,
            circle_of_interest_radius,
            (0, 255, 0),
        )

        feature_of_interest = find_distinctive_feature_coords(gray_img, template_img)
        cv2.circle(visual_rep, feature_of_interest, 3, (0, 0, 255))
        cv2.imshow("image", visual_rep)

        cv2.waitKey(5000)
        cv2.destroyWindow("image")

        # TODO: check for radii that result outside of image
        # Cropping Coordinates

        cropped_image = get_croppedRotated_img(
            img,
            feature_of_interest,
            circle_of_interest_coords,
            circle_of_interest_radius,
            feature_angle,
        )
        cropped_vis = get_croppedRotated_img(
            visual_rep,
            feature_of_interest,
            circle_of_interest_coords,
            circle_of_interest_radius,
            feature_angle,
        )

        # Show Image
        cv2.imshow("image", cropped_vis)
        cv2.waitKey(1)

        # Look for circles to remove
        print("Please select area to remove")
        ignore_points = prompt_for_surrounding_points("image")
        ignore_circle_coords, ignore_circle_radius = find_circle(*ignore_points)
        cv2.circle(cropped_vis, ignore_circle_coords, ignore_circle_radius, (255, 0, 0))
        cv2.imshow("image", cropped_vis)
        cv2.waitKey(1)

        satisfied = input("Is the image satisfactory? (y/n): ") == "y"
        if not satisfied:
            cv2.destroyAllWindows()
        else:
            break

    # Interpolate cropped image to ensure it is (src_height, src_width, src_channels)
    final_image = cv2.resize(
        cropped_image, (src_height, src_width), interpolation=cv2.INTER_LINEAR
    )
    return final_image
