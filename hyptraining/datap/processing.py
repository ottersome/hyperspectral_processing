import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from screeninfo import get_monitors

from ..utils.utils import Point, create_logger

Point = namedtuple("Point", ["x", "y"])


@dataclass
class Circle:
    center: Point
    radius: int


ORIENTATIONS_RAD = [np.pi / 2, np.pi, 3 * np.pi / 2, 0]

logger = create_logger(os.path.basename(__file__))


def get_screen_resolution():
    for m in get_monitors():
        return Point(m.width, m.height)


def resize_image_to_screen(img: np.ndarray) -> Tuple[np.ndarray, float]:
    scrnw, scrnh = get_screen_resolution()
    imgh, imgw = img.shape[:2]
    scaling_factor = min(scrnw / imgw, scrnh / imgh)
    new_width, new_height = (int(imgw * scaling_factor), int(imgh * scaling_factor))
    new_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return new_image, scaling_factor


def rescale_points(original_size: Point, new_size: Point, point: Point):
    scalew, scaley = (new_size.x / original_size.x, new_size.y / original_size.y)
    assert scalew == scaley, "We are not support this"
    new_point = Point(point.x * scalew, point.y * scaley)
    return new_point


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
    cv2.resizeWindow("image")
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
    og_img: np.ndarray, og_feature: np.ndarray
) -> Point:
    assert len(og_img.shape) == 2, f"Input image must be 2d"
    assert len(og_feature.shape) == 2, "Input feature map must be 2D"
    img_gray = og_img.copy().astype(np.float32)
    feature_gray = og_feature.copy().astype(np.float32)
    # Find distinctive feature
    # For now we will use the center of the circle
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # feat_gray = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
    print(f"Image size is {img_gray.shape[0]} ,{img_gray.shape[1]}")
    print(f"Feature size is {feature_gray.shape[0]} ,{feature_gray.shape[1]}")
    feat_width, feat_height = feature_gray.shape[0], feature_gray.shape[1]

    cur_maxval = 0
    cur_loc = [-1, -1]

    print(f"Image we will use is of shape {img_gray.shape}")
    max_vals = []
    max_locs = []
    for r in ORIENTATIONS_RAD:
        score_matrix = cv2.matchTemplate(img_gray, feature_gray, 3)
        rotation = cv2.getRotationMatrix2D(
            (feat_width // 2, feat_height // 2), np.degrees(r), 1
        )
        feature_gray = cv2.warpAffine(feature_gray, rotation, (feat_width, feat_height))
        _, max_val, _, topleft_loc = cv2.minMaxLoc(score_matrix)
        max_vals.append(max_val)
        max_locs.append(topleft_loc)

    # Show the max cal for all orientations
    # print("Max vals")
    # for i, mv in enumerate(max_vals):
    #     print(
    #         f"For orientation {np.degrees(ORIENTATIONS_RAD[i])} max_val {mv} and position{max_locs[i]}"
    #     )

    max_idx = np.argmax(max_vals)
    # print(f"Picking is {max_idx}")
    max_loc = max_locs[max_idx]

    feat_center = Point(
        int(max_loc[1] + feat_width // 2), int(max_loc[0] + feat_height // 2)
    )

    return feat_center


def prompt_for_surrounding_points(image_window_id: str) -> List[Point]:
    """
    WARNING: This will change the image you pass. Please send copy if necessary.
    """
    clicks = []

    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append(Point(x, y))

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
    print("Please wait while image is rotated and cropped")
    rotation_angle = preferred_angle - feature_angle

    rotation_matrix = cv2.getRotationMatrix2D(
        (cropped_height // 2, cropped_width // 2), np.degrees(rotation_angle), 1
    )
    final_image = cv2.warpAffine(
        cropped, rotation_matrix, (cropped_height, cropped_width)
    )

    return final_image


def cropImage(img: np.ndarray, circle_location: Point, circle_radius):
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
    print(f"Using circle location point: {circle_location} with radius _circle_radius")
    cropped = img[
        circle_location.y - circle_radius : circle_location.y + circle_radius,
        circle_location.x - circle_radius : circle_location.x + circle_radius,
        :,
    ]
    return cropped


def rotate_according_to_feature(
    image: np.ndarray, feature_loc: Point, preferred_angle: float
):
    # Angle between args.direction and
    xcenter, ycenter = (image.shape[1] // 2, image.shape[0] // 2)
    feat_from_center = Point(
        feature_loc.x - xcenter,
        -(feature_loc.y - ycenter),
    )
    feature_angle = np.arctan2(feat_from_center.y, feat_from_center.x)
    rotation_angle = preferred_angle - feature_angle

    rotation_matrix = cv2.getRotationMatrix2D(
        (ycenter, xcenter), np.degrees(rotation_angle), 1
    )

    final_image = cv2.warpAffine(
        image, rotation_matrix, (image.shape[0], image.shape[1])
    )

    return final_image


def store_all_channels(img: np.ndarray):
    """For debugging mostly"""
    from pathlib import Path

    place = Path.cwd() / "imglogs/"
    Path(place).mkdir(parents=True, exist_ok=True)
    for c in range(img.shape[2]):
        # Save the cth channel into
        specific_place = Path(place) / f"channel_{c:3d}.png"
        normalized = cv2.normalize(
            img[:, :, c], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        cv2.imwrite(str(specific_place), normalized)


def fix_rotation(choice: int, img: np.ndarray) -> np.ndarray:
    assert choice in [1, 2, 3], "Choice must be 1, 2 or 3"
    center = Point(img.shape[1] // 2, img.shape[0] // 2)
    fix_rotation_matrix = cv2.getRotationMatrix2D(center, 90 * choice, 1)
    final_image = cv2.warpAffine(img, fix_rotation_matrix, (img.shape[0], img.shape[1]))
    return final_image

def dataframe_to_tensor(df: pd.DataFrame, src_width:int, src_height: int ) -> np.ndarray:
    img_columns = df.iloc[:, 2:]
    src_channels = df.shape[1] - 2
    img: np.ndarray = df_to_img(img_columns, src_width, src_height, src_channels)
    return img

def get_standard_source(
    src: pd.DataFrame,
    template_loc: str,
    src_width: int,
    src_height: int,
    src_channels: int,
    feature_angle: float,  # Should be radians
) -> Tuple[np.ndarray, Circle]:
    # Template Image:
    template_img = cv2.imread(template_loc, cv2.IMREAD_GRAYSCALE)
    # Select 2-N columns from dataset
    img = dataframe_to_tensor(src, src_width, src_height)

    final_img = np.ndarray([])
    ignore_spot = Circle(Point(0, 0), 0.0)

    satisfied = False
    while not satisfied:
        # Prompt for points of interest
        # gray_img = np.mean(img, axis=-1) if img.shape[2] > 1 else img  # READ

        cropped_n_rotated_img = get_circle_ofinterest(
            img,  template_img, src_width, src_height, feature_angle
        )

        visual_img = np.stack((cropped_n_rotated_img[:, :, 60],) * 3, axis=-1)
        visual_img = cv2.normalize(
            visual_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        print("Please select area to remove (ignore)")
        cv2.imshow("image", visual_img)
        ignore_points = prompt_for_surrounding_points("image")
        ignore_circle_coords, ignore_circle_radius = find_circle(*ignore_points)
        # TODO: actually ignore the points
        cv2.circle(visual_img, ignore_circle_coords, ignore_circle_radius, (255, 0, 0))
        cv2.imshow("image", visual_img)
        cv2.waitKey(1)

        satisfied = input("Is the image satisfactory? (y/N): ") == "y"
        cv2.destroyAllWindows()
        if satisfied:
            final_img = cv2.resize(
                cropped_n_rotated_img,
                (src_height, src_width),
                interpolation=cv2.INTER_LINEAR,
            )  # CHECK: not sure if we want to interpolate on given data but a shape must be achieved
            print(f"Final image is of shape {final_img.shape}")
            ignore_spot = Circle(ignore_circle_coords, ignore_circle_radius)
            break

    # Interpolate cropped image to ensure it is (src_height, src_width, src_channels)
    return final_img, ignore_spot

def get_circle_ofinterest(
    img: np.ndarray,
    template_img: np.ndarray,
    src_width: int,
    src_height: int,
    feature_angle: float,  # Should be radians
) -> np.ndarray:
    """
    Select circle we are interest in and rotate it as we please
    """
    satisfied = False
    final_img = np.array([])
    while not satisfied:
        gray_img = img[:, :, 60].squeeze() if img.shape[2] > 1 else img  # READ ONLY

        visual_rep = np.stack((gray_img,) * 3, axis=-1).astype(
            np.float32
        )  # Visual changes happen here'
        # Show Image for changes
        cv2.namedWindow("image")
        visual_rep, scaling_factor = resize_image_to_screen(visual_rep)
        visual_rep = cv2.normalize(
            visual_rep, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        # Ask user to select circle within the image
        cv2.imshow("image", visual_rep)
        print("Please select three points to be used to find circle.")
        points_visual = prompt_for_surrounding_points("image")
        cv2.destroyWindow("image")

        print("Using selection points to find circle...")
        # Find Area of Interest
        (
            circle_of_interest_coords_visual,
            circle_of_interest_radius_visual,
        ) = find_circle(*points_visual)
        # Ensure circle is within the image
        if (
            circle_of_interest_coords_visual.x - circle_of_interest_radius_visual < 0
            or circle_of_interest_coords_visual.y - circle_of_interest_radius_visual < 0
            or circle_of_interest_coords_visual.x + circle_of_interest_radius_visual
            > visual_rep.shape[1]
            or circle_of_interest_coords_visual.y + circle_of_interest_radius_visual
            > visual_rep.shape[0]
        ):
            print("Circle is not within the image. Please try again.")
            continue 

        # Tranfer these points to original coordinates
        circle_of_interest_coords_true = Point(
            int(circle_of_interest_coords_visual.x * (1 / scaling_factor)),
            int(circle_of_interest_coords_visual.y * (1 / scaling_factor)),
        )
        circle_of_interest_radius_true = int(
            circle_of_interest_radius_visual * (1 / scaling_factor)
        )

        # Now Crop
        print("Cropping image around the diameter of the circle.")
        cropped_img_true = cropImage(
            img, circle_of_interest_coords_true, circle_of_interest_radius_true
        )
        print("Automatically finding the features for alignment.")

        # Feature of interest
        feature_of_interest = find_distinctive_feature_coords(
            cropped_img_true[:, :, 60], template_img
        )
        # Rotate the Image(according to feature of interest)
        print(
            f"Feature found at {np.degrees(feature_angle)} degrees. Aligning image according to feature..."
        )
        rotated_img = rotate_according_to_feature(
            cropped_img_true, feature_of_interest, feature_angle
        )
        cropped_visual = np.stack((cropped_img_true[:, :, 60].copy(),) * 3, axis=-1)
        cropped_visual = cv2.normalize(  # type: ignore
            cropped_visual, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U  # type : ignore
        )
        visual_img = np.stack((rotated_img[:, :, 60].copy(),) * 3, axis=-1)
        visual_img = cv2.normalize(  # type: ignore
            visual_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U  # type : ignore
        )
        # Prompt user for corrections
        # Show original and rotated for comparison
        two_images = np.concatenate((cropped_visual, visual_img), axis=1)
        print(f"Channels of concatenated images {two_images.shape}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(two_images, "Original", ( 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(two_images, "After Rotation", (10 + visual_img.shape[1], 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("image", two_images)
        print(
            "Cropping and feature finding has been executed. Please select:\n"
            "1) Rotate the image 90 degrees\n"
            "2) Rotate the image 180 degrees\n"
            "3) Rotate the image 270 degrees\n"
            "4) Continue"
        )
        cv2.waitKey(100)
        decision = input("Your choice (and Enter): ")
        cv2.destroyAllWindows()
        satisfied = True
        if decision in ["1", "2", "3"]:
            decision = int(decision)
            print(f"Rotating {90*decision} degrees...")
            rotated_img = fix_rotation(decision, cropped_img_true)
            visual_img = np.stack((rotated_img[:, :, 60].copy(),) * 3, axis=-1)
            visual_img = cv2.normalize(  # type: ignore
                visual_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U  # type : ignore
            )    
        else:
            print("Continuing")
        final_img = rotated_img
    return final_img
