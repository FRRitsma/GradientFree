# %%
from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import prod
import cvxpy as cp
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def rastrigin(*X, **kwargs):
    
    A = kwargs.get("A", 10)
    return A + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])


@dataclass
class PolynomialBundle:
    pass




@dataclass
class Polynomial:
    coefficients: np.ndarray = None
    n_parameters: int = None
    pca: PCA = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ):
        # Function scope hardcode:
        max_parameters = max(X.shape)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=0.2,
        )

        # Get best performance:
        best_score = float("inf")
        best_parameters = 0
        for i in range(2, max_parameters):
            self.__fit(X_train, Y_train, i)
            Y_pred = self.predict(X_test)
            new_score = mean_squared_error(Y_test, Y_pred)

            if new_score < best_score:
                best_score = new_score
                best_parameters = i

        self.__fit(X_train, Y_train, best_parameters)

    def __fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        parameters: int,
    ) -> None:

        # Fit pca for rank insufficient data:
        pca = PCA().fit(X)
        X = pca.transform(X)

        # Components of the polynomial:
        exponents = form_exponents(X, parameters)
        constants = cp.Variable(exponents.shape[1])

        # Formalization of optimization problem:
        objective = cp.Minimize(
            0.5
            * cp.sum_squares(
                exponents @ constants - Y.ravel(),
            )
        )
        prob = cp.Problem(objective)

        # Try different solvers:
        try:
            prob.solve(solver=cp.ECOS)
        except cp.error.SolverError:
            prob.solve(solver=cp.SCS)

        # Assign results to self:
        self.n_parameters = parameters
        self.coefficients = constants.value
        self.pca = pca

    def predict(self, X) -> np.ndarray:
        # Ensure that all relevant components have been initialized:
        assert (
            self.n_parameters is not None
            and self.coefficients is not None
            and self.pca is not None
        ), "Polynomial object has not been fitted"

        X = self.pca.transform(X)

        exponents = form_exponents(X, self.n_parameters)

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





def form_exponents(x: np.ndarray, parameters: int) -> np.ndarray:
    assert len(x.shape) == 2, "Input x must be two dimensional."
    assert parameters > 0, "Parameters argument must be larger than 0"

    exponents = np.empty([x.shape[0], parameters])

    x = np.hstack([np.ones((x.shape[0], 1)), x])

    for i, vars in enumerate(combinations_with_replacement(x.T, 10)):
        exponents[:, i] = prod(vars)
        if i + 1 >= parameters:
            break
    return exponents


# Creating fit

x = np.random.rand(80, 2)
y = 2 * x[:, 0] ** 2 + 3 * x[:, 1] ** 2 + 2 * x[:, 0] + 3

# %%

poly = Polynomial()
poly.fit(x, y)
poly.__repr__()
# print(x)
# print(form_exponents(x, 4))
# print(poly.predict(x), y)
# Testing

# %%
# MEMORY = 8
# n = 3
# X = 4 * np.random.rand(1, 2) - np.array([2, 1])
# X = X + float(1e-1) * np.random.rand(n, 2)
# # Y = rosenbrock_np(X)
# Y = rastrigin(X[:, 0], X[:, 1])

# plt.scatter(X[:, 0], X[:, 1], c=Y)
# plt.show()
# print(Y[-MEMORY:])

# for i in range(1000):
#     pol = Polynomial()
#     pol.fit(X, Y, degree=max_degree(X.shape))
#     x_best = find_optimal_point(
#         X[-MEMORY:, :],
#         pol,
#         step_size=1,
#         n_samples=int(1e5),
#     )

#     X = np.vstack([X, x_best])
#     # X[-MEMORY:, :] = variance_correct(X[-MEMORY:, :])

#     # Y = np.hstack([Y, rosenbrock_np(X[-1, :])])
#     Y = np.hstack([Y, rastrigin(X[-1, 0], X[-1, 1])])
#     time.sleep(0.5)

#     plt.scatter(X[:, 0], X[:, 1], c=Y)
#     plt.show()
#     print(Y[-1])
