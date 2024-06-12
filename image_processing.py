"""
This is a file used for experimetns. Not really used for much but for education
"""

import argparse
from typing import Tuple

import cv2
import numpy as np

from hyptraining.utils.utils import Point

# Extend Point to take in an ndarray for construction

ORIENTATIONS = ["top", "left", "bottom", "right"]
ORIENTATIONS_RAD = [np.pi / 2, np.pi, 3 * np.pi / 2, 0]
OR_TO_RAD = {k: v for k, v in zip(ORIENTATIONS, ORIENTATIONS_RAD)}


def arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="./pacman.png")
    ap.add_argument("--feature", default="./feature_right.png")
    ap.add_argument(
        "--orientation",
        default="top",
        choices=ORIENTATIONS,
        help="Direction of feature",
    )
    return ap.parse_args()


def circleRadius(b, c, d) -> Tuple[Point, int]:
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


def draw_result(
    circle_point: Point,
    radius,
    image: np.ndarray,
    feature_point: Point,
):
    # Draw the selected circle
    int_points = (int(circle_point[0]), int(circle_point[1]))
    cv2.circle(image, int_points, int(radius), (0, 255, 0), 2)
    cv2.circle(image, (int(feature_point[0]), int(feature_point[1])), 5, (0, 0, 255), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)


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
    coords, circle_radius = circleRadius(*clicks)

    return coords, circle_radius


# TODO: find a distinctive feature to base rotation from
def find_distinctive_feature_coords(img: np.ndarray, feature: np.ndarray) -> Point:
    # Find distinctive feature
    # For now we will use the center of the circle
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat_gray = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
    print(f"Image size is {img_gray.shape[0]} ,{img_gray.shape[1]}")
    print(f"Feature size is {feature.shape[0]} ,{feature.shape[1]}")
    feat_width, feat_height = feature.shape[0], feature.shape[1]

    cur_maxval = 0
    cur_loc = [-1, -1]

    for r in ORIENTATIONS_RAD:
        score_matrix = cv2.matchTemplate(img_gray, feat_gray, 3)
        rotation = cv2.getRotationMatrix2D(
            (feat_width // 2, feat_height // 2), np.degrees(r), 1
        )
        feat_gray = cv2.warpAffine(feat_gray, rotation, (feat_width, feat_height))
        _, max_val, _, topleft_loc = cv2.minMaxLoc(score_matrix)
        if max_val > cur_maxval:
            cur_maxval = max_val
            cur_loc = topleft_loc

    assert cur_loc[0] != -1, cur_loc[1] != -1

    feat_center = Point(cur_loc[0] + feat_width // 2, cur_loc[1] + feat_height // 2)

    return feat_center


def get_final_image(
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
        feature_location.x - circle_location.x + cropped_width // 2,
        feature_location.y - circle_location.y + cropped_height // 2,
    )
    # Ensure feature within radius
    assert (
        feature_local.x < cropped_width and feature_local.y < cropped_height
    ), "Feature not within circle"
    # Angle between args.direction and
    feat_from_center = Point(
        feature_local.x - cropped_width // 2,
        feature_local.y - cropped_height // 2,
    )
    print(f"Feature from center {feat_from_center}")
    feature_angle = np.arctan2(-feat_from_center.y, feat_from_center.x)
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

    # Draw Image and then add feature location
    cv2.circle(cropped, feature_local, 5, (0, 255, 0), 2)
    cv2.circle(cropped, (cropped_width // 2, cropped_height // 2), 5, (255, 0, 0), 2)
    cv2.imshow("Test", cropped)
    cv2.waitKey(0)

    return final_image


"""
As an example pipline:
"""


def example(args: argparse.Namespace):
    # Load the image
    img: np.ndarray = cv2.imread(args.img)
    # Get size along 3 dimensions
    h, w, c = img.shape
    print(f"Image Height {h}, Width {w}, channels {c}")
    # To numpy format
    img = np.array(img)
    # Get the Template Image
    template_image = cv2.imread(args.feature)

    circle_coords, radius = chords_to_circle(img)

    feat_center_coords = find_distinctive_feature_coords(img, template_image)

    preferred_angle = OR_TO_RAD[args.orientation]

    final_image = get_final_image(
        img, feat_center_coords, circle_coords, radius, preferred_angle
    )
    cv2.imshow("Final fixed image", final_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    args = arguments()
    example(args)
