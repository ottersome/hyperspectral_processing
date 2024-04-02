import argparse
import tkinter as tk
from typing import Tuple

import cv2
import numpy as np


def arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="./pacman.png")
    return ap.parse_args()


def circleRadius(b, c, d):
    B = np.array(
        [
            [(c[0] ** 2 + c[1] ** 2 - b[0] ** 2 - b[1] ** 2) / 2],
            [(d[0] ** 2 + d[1] ** 2 - c[0] ** 2 - c[1] ** 2) / 2],
        ]
    )

    A = np.array([[c[0] - b[0], c[1] - b[1]], [d[0] - c[0], d[1] - c[1]]])
    circle_center = np.linalg.inv(A).dot(B)
    print(f"Circle center is {circle_center}")
    return (
        np.sqrt((circle_center[0] - b[0]) ** 2 + (circle_center[1] - b[1]) ** 2),
        circle_center,
    )


def draw_result(point: Tuple[int, int], radius, image: np.ndarray):
    # Draw the selected circle
    int_points = (int(point[0]), int(point[1]))
    cv2.circle(image, int_points, int(radius), (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)


def select_three_chord_ends(img: np.ndarray):
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
    cv2.imshow("image", img)
    # Quit only after three clicks
    while len(clicks) < 3:
        cv2.waitKey(1)

    # Once the windows are distroyed
    print("Obtaining circle radius")
    circle_radius, coords = circleRadius(*clicks)
    print("Drawing result")
    draw_result(coords, circle_radius, img)


# TODO: find a distinctive feature to base rotation from


def main(args: argparse.Namespace):
    # Load the image
    img = cv2.imread(args.img)
    # Get size along 3 dimensions
    # h, w, c = img.shape
    # To numpy format
    img = np.array(img)
    select_three_chord_ends(img)


if __name__ == "__main__":
    args = arguments()
    main(args)
