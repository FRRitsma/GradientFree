# %%
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import prod
import time
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import math


def rastrigin(*X, **kwargs):
    A = kwargs.get("A", 10)
    return A + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])


@dataclass
class Polynomial:
    coefficients: np.ndarray = None
    degree: int = None

    def fit(self, X: np.ndarray, Y: np.ndarray, degree: int) -> None:
        exponents = form_exponents(X, degree)
        constants = cp.Variable(exponents.shape[1])
        objective = cp.Minimize(
            0.5
            * cp.sum_squares(
                exponents @ constants - Y.ravel(),
            )
        )
        prob = cp.Problem(objective)

        try:
            prob.solve(solver=cp.ECOS)
        except cp.error.SolverError:
            prob.solve(solver=cp.SCS)

        # Assign to self:
        self.degree = degree
        self.coefficients = constants.value

    def predict(self, X) -> np.ndarray:
        assert (
            self.degree is not None and self.coefficients is not None
        ), "Polynomial object has not been fitted"
        exponents = form_exponents(X, self.degree)
        return exponents @ self.coefficients


def find_optimal_point(
    x: np.ndarray,
    pol: Polynomial,
    step_size: float = float(1e-1),
    n_samples: int = int(2e4),
) -> np.ndarray:

    # Settings:
    search_range_min = np.min(x, axis=0) - step_size
    search_range_max = np.max(x, axis=0) + step_size

    # Generate samples:
    samples = search_range_min + np.random.rand(n_samples, x.shape[1]) * (
        search_range_max - search_range_min
    )

    # Filter by proximity:
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(x)
    distance, _ = knn.kneighbors(samples)
    samples = samples[np.squeeze(distance) < step_size, :]

    # Get scores:
    scores = pol.predict(samples)

    # Get best sample:
    best_sample = samples[scores == min(scores), :]

    return best_sample


def rosenbrock_np(xy: np.ndarray) -> np.ndarray:
    xy = xy.reshape(-1, 2)
    return rosenbrock(xy[:, 0], xy[:, 1])


def rosenbrock(x: float, y: float) -> float:
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x**2) ** 2


def parameters_dim_degree(dim: int, degree: int) -> int:
    """How many parameters are needed to fit a polynomial of given degree
    and dimension

    Args:
        dim (int): Dimension of input data
        degree (int): Degree of polynomial

    Returns:
        int: Amount of parameters needed to form polynomial function
    """
    n_parameters = sum(
        1 for _ in combinations_with_replacement(list(range(dim + 1)), degree)
    )
    return n_parameters


def max_degree(shape: tuple):
    assert type(shape) is tuple, "Input must be shapelike"
    assert len(shape) == 2, "Data must be two dimensional"

    n_samples = shape[0]
    dim = shape[1]

    for degree in range(n_samples):
        if parameters_dim_degree(dim, degree) > n_samples:
            break
    degree = max(degree - 1, 0)

    return degree


def fill_row_with_variables(x, degree: int = 2):
    dim = len(x)
    x = np.hstack([np.squeeze(x), [1]])
    a = np.empty([parameters_dim_degree(dim, degree)])
    for i, vars in enumerate(combinations_with_replacement(x, degree)):
        a[i] = prod(vars)
    return a


def form_exponents(x: np.ndarray, degree: int = 2) -> np.ndarray:
    assert len(x.shape) == 2, "Input x must be two dimensional."
    assert degree > -1, "Degree must be larger than zero"
    X = np.empty([x.shape[0], parameters_dim_degree(x.shape[1], degree)])

    # for index, row in enumerate(x):
    #     X[index, :] = fill_row_with_variables(row, degree)

    x = np.hstack([x, np.ones((x.shape[0], 1))])
    for i, vars in enumerate(combinations_with_replacement(x.T, degree)):
        X[:, i] = prod(vars)

    return X


# Creating fit

x = np.random.rand(20, 2)
y = 2 * x[:, 0] ** 2 + 3 * x[:, 1] ** 2 + 2 * x[:, 0] + 3

poly = Polynomial()
poly.fit(x, y, degree=max_degree(x.shape))

print(form_exponents(x, degree=max_degree(x.shape)))
# print(poly.predict(x), y)
# Testing

# %%
MEMORY = 8
n = 3
X = 4 * np.random.rand(1, 2) - np.array([2, 1])
X = X + float(1e-1) * np.random.rand(n, 2)
# Y = rosenbrock_np(X)
Y = rastrigin(X[:, 0], X[:, 1])

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.show()
print(Y[-MEMORY:])

for i in range(1000):
    pol = Polynomial()
    pol.fit(X, Y, degree=max_degree(X.shape))
    x_best = find_optimal_point(
        X[-MEMORY:, :],
        pol,
        step_size=1,
        n_samples=int(1e5),
    )

    X = np.vstack([X, x_best])
    # X[-MEMORY:, :] = variance_correct(X[-MEMORY:, :])

    # Y = np.hstack([Y, rosenbrock_np(X[-1, :])])
    Y = np.hstack([Y, rastrigin(X[-1, 0], X[-1, 1])])
    time.sleep(0.5)

    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()
    print(Y[-1])
