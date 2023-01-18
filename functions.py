# %%
import numpy as np
import cvcx



def rosenbrock_np(xy: np.ndarray) -> np.ndarray:
    xy = xy.reshape(-1, 2)
    return rosenbrock(xy[:, 0], xy[:, 1])


def rosenbrock(x: float, y: float) -> float:
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x**2) ** 2


def triangle(m: int) -> int:
    """Generate the triangle number, i.e.: 1 + 2 + ... + m for input m

    Args:
        m (int): End of sequence

    Returns:
        int: Sum of integers until m
    """
    return int((m * (m + 1)) / 2)


def parameter_count(dimension: int) -> int:
    """Amount of parameters for a polynomial of given dimension

    Args:
        dimension (int): Dimension of quadratic polynomial

    Returns:
        int: Amount of parameters required for quadratic polynomial in given
        dimension
    """
    return 1 + dimension + triangle(dimension)


def max_dimension_count(no_samples: int) -> int:
    """The maximum dimension for which a quadratic function can be fitted
    for the given amount of samples

    Args:
        no_samples (int): Amount of samples of function to be fitted

    Returns:
        int: Maximum dimension for which sufficient data exists
    """
    # Function scope hardcode:
    MAX_DIMENSION = 1000
    # Return if requried parameters exceed amount of samples:
    for i in range(MAX_DIMENSION):
        if parameter_count(i) > no_samples:
            return i - 1
    return i


def inverse_triangle(m: int) -> int:
    return int((2 * m + 1 / 4) ** (1 / 2) - 1 / 2)


# %%
