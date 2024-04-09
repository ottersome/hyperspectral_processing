import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_t

from .utils import Point


def sample_distribution_per_pixel(
    i: int,
    j: int,
    center: Point,
    radius: float,
    scale: float,
    offset: float = 0.5,
    dropoff_distance: float = 50,
):
    """
    Returns sample from a distribution that depends on i,j.
    Normal distribution. The further away from center the higher the mean.
    Ideally we would sample what look like a mountain

    Perhaps we can reduce complexity by calculating 4 samples (as if in a single quadrant)
    and then mirroring them
    """
    dropoff_distance /= 2
    distance_to_1 = 0.01

    # Calculate distance from center
    y = i - center.y
    x = -1 * (j - center.x)

    distance = np.sqrt(x**2 + y**2)

    # Calculate mean
    # mean = 1 + distance / 10
    b = (1 / dropoff_distance) * np.log((1 - distance_to_1) / distance_to_1)

    # Should give us a nice dropopp
    difference = 1 - 1 / (
        1
        + np.exp(
            -b
            # Min is being used to avoid exponential overflow
            * (min(distance, radius + dropoff_distance * 2) - radius - dropoff_distance)
        )
    )
    mean = offset * difference

    # Use something like Relu to have a cuttoff at radiu

    sample = np.random.normal(loc=mean, scale=scale, size=1).item()
    sample_clipped = np.clip(sample, 0, 1)
    return sample_clipped


def student_t_show(mean: np.ndarray, cov: np.ndarray):
    # Parameters for the bivariate t-distribution
    df = 10  # Degrees of freedom
    # Create a multivariate t-distribution object
    rv = multivariate_t(mean, cov, df)  # type:ignore
    # Generate a grid of points where we want to evaluate the PDF
    x, y = np.mgrid[0:5:0.01, 0:5:0.01]
    pos = np.dstack((x, y))
    # Evaluate the PDF at the grid points
    pdf = rv.pdf(pos)
    # Plot the PDF using a contour plot
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, pdf, cmap="viridis")
    plt.colorbar(label="Probability Density")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bivariate t-distribution PDF")
    plt.show()


def sample_studentt(mean: np.ndarray, cov: np.ndarray):
    # Parameters for the bivariate t-distribution
    cov = np.array([[10, 0.1], [0.1, 10]])
    df = 10  # Degrees of freedom
    # Create a multivariate t-distribution object
    rv = multivariate_t(mean, cov, df)  # type:ignore
    # Sample from the distribution
    samples = rv.rvs(1000)
    # Plot the samples
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bivariate t-distribution Samples")
    plt.show()
    exit()
