import logging
import os
from collections import namedtuple

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


# Make a decorator called `unused` this will wrap a function and raise an exception whenver its caled because its being faded out of development
def unused(func):
    def wrapper(*args, **kwargs):
        raise Exception(f"{func.__name__} is no longer in use")

    return wrapper
