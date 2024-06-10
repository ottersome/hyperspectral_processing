import logging
import os
from collections import namedtuple
import cv2
import numpy as np

Point = namedtuple("Point", ["x", "y"])


def create_logger(name: str):
    logger = logging.getLogger(name)
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


def dump_2dimg_to_log(array: np.ndarray, array_name: str, file: str):
    with open(file, "w+") as f:
        f.write(f"Start dump of {array_name}--------------------\n")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                f.write(f"{array[i,j]:.2f}, ")
            f.write("\n")
        f.write(f"End dump of {array_name}-------------------\n")


def dump_3dimg_to_log(array: np.ndarray, array_name: str, file: str):
    with open(file, "w+") as f:
        f.write(f"Start dump of {array_name}--------------------\n")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                f.write("( ")
                for k in range(array.shape[2]):
                    f.write(f"{array[i,j,k]:.2f}, ")
                f.write(" )")
            f.write("\n")
        f.write(f"End dump of {array_name}-------------------\n")


def normalize_3d_image(tensor: np.ndarray):
    for i in range(tensor.shape[2]):
        tensor[:, :, i] = cv2.normalize(
            tensor[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
    return tensor


# Make a decorator called `unused` this will wrap a function and raise an exception whenver its caled because its being faded out of development
def unused(func):
    def wrapper(*args, **kwargs):
        raise Exception(f"{func.__name__} is no longer in use")

    return wrapper


def compare_picture(estimation: np.ndarray, ground_truth: np.ndarray):
    """
    Will plot both images for comparison
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axs[0].imshow(estimation)
    axs[1].imshow(ground_truth)
    plt.show()
    return


def draw_point_in_image(image: np.ndarray, point: Point):
    """
    Generally used for debugging the program.
    Will show the image at hand with a red dot at the point passed
    """
    if image.shape[2] > 3 or image.shape[2] == 2:
        image = image.mean(axis=-1)
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 1:
        image = image.mean(axis=-1)

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.circle(image, (point.x, point.y), 4, (0, 0, 255), -1)
    # Show it it
    cv2.imshow("Image", image)
    # Wait for "q" to be pressed
    key = cv2.waitKey(0)
    while key & 0xFF != ord("q"):
        key = cv2.waitKey(0)
    cv2.destroyAllWindows()
