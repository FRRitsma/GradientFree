# %%
import cvxpy as cp
import numpy as np
from functions import parameter_count, triangle
from sklearn.neighbors import NearestNeighbors
from itertools import combinations_with_replacement
from functions import rosenbrock_np
from development import variance_correct
import time
from math import prod


def demo():
    A = np.array(
        [[1, 19039], [1, 47792], [1, 9672], [1, 32521], [1, 11409], [1, 58843]]
    )
    b = np.array(
        [
            [0.05938044],
            [0.27213514],
            [0.00252875],
            [0.18535543],
            [0.01959069],
            [0.52605937],
        ]
    )
    b = np.squeeze(b)
    C_min = np.array([[-1, 0]])
    C_max = np.array([[1, 65535]])

    x = cp.Variable(A.shape[1])

    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b))
    constraints = [C_min @ x <= 0, C_max @ x <= 1]

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.ECOS)

    intercept = x.value[0]
    slope = x.value[1]
    return result, intercept, slope


def parameters_dim_degree(dim: int, degree: int) -> int:
    n_parameters = sum(
        1 for _ in combinations_with_replacement(list(range(dim + 1)), degree)
    )
    return n_parameters


def fill_row_with_variables_v2(x, degree: int = 2):
    dim = len(x)
    x = np.hstack([np.squeeze(x), [1]])
    a = np.empty([parameters_dim_degree(dim, degree)])
    for i, vars in enumerate(combinations_with_replacement(x, degree)):
        a[i] = prod(vars)
    return a


def fill_row_with_variables(x, degree: int = 2):
    dim = len(x)
    a = np.empty([parameter_count(dim)])
    for i, (x0, x1) in enumerate(combinations_with_replacement(x, degree)):
        a[i] = x0 * x1
    for i, x0 in enumerate(x):
        a[i + triangle(dim)] = x0
    a[-1] = 1
    return a


def form_exponents(x: np.ndarray, degree: int = 2):
    assert len(x.shape) == 2, "Input x must be two dimensional."
    X = np.empty([x.shape[0], parameters_dim_degree(x.shape[1], degree)])

    for index, row in enumerate(x):
        X[index, :] = fill_row_with_variables_v2(row, degree)

    return X


def fit_exponential(x: np.ndarray, y: np.ndarray):
    exponents = form_exponents(x)
    constants = cp.Variable(exponents.shape[1])
    objective = cp.Minimize(0.5 * cp.sum_squares(exponents @ constants - y))
    prob = cp.Problem(objective)
    try:
        prob.solve(solver=cp.ECOS)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS)

    return constants.value


def output_polynomial(x: np.ndarray, c: np.ndarray):
    exponents = form_exponents(x)
    return exponents @ c


def find_optimal_point(
    x: np.ndarray,
    c: np.ndarray,
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
    scores = output_polynomial(samples, c)

    # Get best sample:
    best_sample = samples[scores == min(scores), :]

    return best_sample


# Perform optimization:
x = np.random.rand(20, 2)
y = 2 * x[:, 0] ** 2 + 3 * x[:, 1] ** 2 + 2 * x[:, 0] + 3
c = fit_exponential(x, y)
x_best = find_optimal_point(x, c)


# %%

if __name__ == "__main__":
    #  Initialize iteration points:

    # %%
    import matplotlib.pyplot as plt

    MEMORY = 8
    n = 6
    X = 4 * np.random.rand(1, 2) - np.array([2, 1])
    X = X + float(1e-1) * np.random.rand(n, 2)
    Y = rosenbrock_np(X)

    # %%
    plt.scatter(X[-MEMORY:, 0], X[-MEMORY:, 1], c=Y[-MEMORY:])
    plt.show()
    print(Y[-MEMORY:])

    for i in range(1000):
        c = fit_exponential(X[-MEMORY:, :], Y[-MEMORY:])
        x_best = find_optimal_point(
            X[-MEMORY:, :], c, step_size=0.1, n_samples=int(1e5)
        )

        X = np.vstack([X, x_best])
        # X[-MEMORY:, :] = variance_correct(X[-MEMORY:, :])

        Y = np.hstack([Y, rosenbrock_np(X[-1, :])])
        time.sleep(0.1)

        plt.scatter(X[:, 0], X[:, 1], c=Y)
        plt.show()
