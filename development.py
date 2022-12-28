# %%
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


def fit_quadratic_2(
    X,  # Recorded input of iterations
    Y,  # Recorded output of iterations
    minimum: float = 0,
    # Initial guesses:
    init_X0: torch.Tensor = None,
    init_V: torch.Tensor = None,
    init_C: torch.Tensor = None,
):
    dim = X.shape[1]
    X = torch.tensor(X)

    # Function scope hardcode:
    LEARNING_RATE = float(1e-1)
    ITERATIONS = int(1e4)

    # Initialize components of function:
    X0 = (
        X[Y == np.min(Y), :].ravel().to(torch.float64)
        if init_X0 is None
        else torch.Tensor(init_X0).to(torch.float64)
    )
    V = (
        torch.rand(dim, dim).to(torch.float64)
        if init_V is None
        else torch.Tensor(init_V).to(torch.float64)
    )
    C = (
        torch.Tensor([np.min(Y)]).to(torch.float64)
        if init_C is None
        else torch.Tensor(init_C).to(torch.float64)
    )

    V.requires_grad = True
    X0.requires_grad = True
    C.requires_grad = True

    optimizer = torch.optim.Adam([V, X0, C], lr=LEARNING_RATE)

    # Fit parameters to data:
    for _ in range(ITERATIONS):
        # Decrease learning rate over time:
        if _ % 1000 == 0:
            LEARNING_RATE = max(LEARNING_RATE / 10, float(1e-5))
        optimizer.zero_grad()
        loss = 0
        for x, y in zip(X, Y):
            p = (
                (x - X0) @ V @ V.T @ (x.T - X0.T)
                + torch.abs(C)
                + torch.Tensor([minimum])
            )
            loss = loss + torch.abs(p - y)
        loss.backward()
        optimizer.step()

    # Convert types to numpy for output:
    X0 = X0.detach().numpy()
    V = V.detach().numpy()
    C = torch.abs(C).detach().numpy() + minimum
    return X0, V, C


class PolynomialFit:
    def fit(self, X, Y):
        self.pca = PCA(whiten=True).fit(X)
        # Transform data with pca:
        Xt = self.pca.transform(X)
        # Fit polynomial to data:
        X0, V, C = fit_quadratic_2(Xt, Y)

        return self.pca.inverse_transform(X0)


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


# %%
if __name__ == "__main__":
    # %% Initialize iteration points:
    MEMORY = 20
    n = 6
    X = 4 * np.random.rand(1, 2) - np.array([2, 1])
    X = X + float(1e-3) * np.random.rand(n, 2)
    Y = rosenbrock_np(X)

    # %%
    for i in range(50):
        polyfit = PolynomialFit()
        fit_x0 = polyfit.fit(X[-MEMORY:, :], Y[-MEMORY:])

        X = np.vstack([X, fit_x0])
        X[-MEMORY:, :] = variance_correct(X[-MEMORY:, :])
        Y = np.hstack([Y, rosenbrock_np(fit_x0)])
        plt.scatter(X[-MEMORY:, 0], X[-MEMORY:, 1])
        plt.show()
        print(Y[-MEMORY:])

# %% Development of function:
# xsub = X[-8:, :]
VARIANCE_FRACTION = 0.1
