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


def variance_alarm(X):
    # Hardcode:
    VARIANCE_FRACTION = 0.1

    # Fit pca to noise
    x_t = PCA().fit_transform(X)
    var = np.var(x_t, axis=0)
    if np.min(var) / np.max(var) < VARIANCE_FRACTION:
        return False
    return True


def variance_correct(X):
    # Hardcode
    # Axes must have variance no less then "VARIANCE_FRACTION" of the maximum:
    VARIANCE_FRACTION = 0.1

    # Rotate axes:
    pca = PCA()
    pca.fit(X)
    T = pca.transform(X)

    # Investigate the biggest variance:
    var = np.var(T, axis=0)
    mean = np.mean(T, axis=0)
    row = T[-1, :]
    N = T.shape[0]

    # Correct where variance is lacking:
    for i, (v, m) in enumerate(zip(var, mean)):
        if v < np.max(var) * VARIANCE_FRACTION:
            print("CORRECTION")
            # Remove variance added by last row:
            last_row_var = ((row[i] - m) ** 2) / N
            # Corrected variance:
            corrected_var = v - last_row_var
            # How much variance must be added by last row:
            target_var = np.max(var) * VARIANCE_FRACTION - corrected_var
            # How much distance:
            target_distance = (target_var * N) ** (1 / 2)
            # Decrease distance from mean, not sign:
            sign = np.sign(row[i] - m)
            T[-1, i] = m + target_distance * sign

    # Reverse transform and return:
    return pca.inverse_transform(T)
