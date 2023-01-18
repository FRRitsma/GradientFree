# %%
import numpy as np
import torch
from sklearn.decomposition import PCA


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


def generate_parameters(X, Y):

    no_samples = X.shape[0]
    dim = X.shape[1]

    # Declaring quadratic:
    used_samples = max_dimension_count(no_samples)
    used_samples = min(used_samples, dim)
    X0 = X[Y == np.min(Y), :]
    X0 = X0[0, :]
    X0 = X0.ravel().to(torch.float64)
    V = torch.rand(dim, used_samples).to(torch.float64)
    C = torch.Tensor([0]).to(torch.float64)

    V.requires_grad = True
    X0.requires_grad = True
    C.requires_grad = True
    no_samples = no_samples - used_samples

    if no_samples < 3:
        return X0, V, C

    used_samples = max_dimension_count(no_samples)
    used_samples = min(used_samples, dim)
    X0_1 = X[Y == np.min(Y), :]
    X0_1 = X0_1[0, :]
    X0_1 = X0_1.ravel().to(torch.float64)
    V_1 = torch.rand(dim, used_samples).to(torch.float64)
    C_1 = torch.Tensor([0]).to(torch.float64)

    V_1.requires_grad = True
    X0_1.requires_grad = True
    C_1.requires_grad = True

    return X0, V, C, X0_1, V_1, C_1


def loss_function_2(X, Y, X0, V, C):
    loss = 0
    for x, y in zip(X, Y):
        p = (x - X0) @ V @ V.T @ (x.T - X0.T) + torch.abs(C)
        loss = loss + torch.abs(p - y)
    return loss


# (x-X0)*A*(x-X0)*(x*B*x + Ct*x + D) +


def loss_function(X, Y, params):
    loss = 0
    X0 = params[0]
    V = params[1]
    C = params[2]

    for x, y in zip(X, Y):
        p = (x - X0) @ V @ V.T @ (x.T - X0.T)
        if len(params) > 3:
            X0_1 = params[3]
            V_1 = params[4]
            C_1 = params[5]
            p = p * ((x - X0_1) @ V_1 @ V_1.T @ (x.T - X0_1.T) + C_1**2)
        p = p + C**2
        loss = loss + torch.abs(p - y)
    return loss


def fit_quadratic_2(
    X: torch.Tensor,  # Recorded input of iterations
    Y: torch.Tensor,  # Recorded output of iterations
    # Initial guesses:
    init_X0: torch.Tensor = None,
    init_V: torch.Tensor = None,
    init_C: torch.Tensor = None,
    # Learning rate:
    LEARNING_RATE: float = float(1e-3),
):
    X = torch.tensor(X)

    # Function scope hardcode:
    # LEARNING_RATE = float(1e-2)
    ITERATIONS = int(1e4)  # int(5e4)

    params = generate_parameters(X, Y)

    # Initializing optimizer
    optimizer = torch.optim.LBFGS(
        params,
        lr=LEARNING_RATE,
        max_iter=ITERATIONS,
    )

    # Pre-optimization loss:
    pre_loss = loss_function(X, Y, params).detach()

    # Fit parameters to data:
    def closure():
        optimizer.zero_grad()
        loss = loss_function(X, Y, params)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Post-optimization loss:
    post_loss = loss_function(X, Y, params).detach()

    if post_loss > pre_loss:
        print("NUMERICAL INSTABILITY")
        return fit_quadratic_2(X, Y, LEARNING_RATE=LEARNING_RATE / 10)

    X0 = params[0]
    V = params[1]
    C = params[2]

    # Convert types to numpy for output:
    X0 = X0.detach().numpy()
    V = V.detach().numpy()
    C = torch.abs(C).detach().numpy()

    if np.any(np.isnan(X0)):
        print("NUMERICAL INSTABILITY")
        return fit_quadratic_2(X, Y, LEARNING_RATE=LEARNING_RATE / 10)

    return X0, V, C


class PolynomialFit:
    def fit(self, X, Y, optim_newton=False):
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
    MEMORY = 40
    n = 3
    X = 4 * np.random.rand(1, 2) - np.array([2, 1])
    X = X + float(1e-1) * np.random.rand(n, 2)
    Y = rosenbrock_np(X)

    # %%
    import matplotlib.pyplot as plt

    for i in range(1000):
        # polyfit = PolynomialFit()
        fit_x0, vs, _ = fit_quadratic_2(X[-MEMORY:, :], Y[-MEMORY:])
        X = np.vstack([X, fit_x0])
        X[-MEMORY:, :] = variance_correct(X[-MEMORY:, :])
        Y = np.hstack([Y, rosenbrock_np(fit_x0)])
        plt.scatter(X[-MEMORY:, 0], X[-MEMORY:, 1], c=Y[-MEMORY:])
        plt.show()
        print(Y[-MEMORY:])


# %% Development of function:
# xsub = X[-8:, :]
VARIANCE_FRACTION = 0.1
