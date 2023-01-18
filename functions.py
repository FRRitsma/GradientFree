# %%
import numpy as np
import torch


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


# def fit_quadratic_2(
#     X,  # Recorded input of iterations
#     Y,  # Recorded output of iterations
#     minimum: float = 0,
#     # Initial guesses:
#     init_X0: torch.Tensor = None,
#     init_V: torch.Tensor = None,
#     init_C: torch.Tensor = None,
# ):
#     dim = X.shape[1]
#     X = torch.tensor(X)

#     # Function scope hardcode:
#     LEARNING_RATE = float(1e-1)
#     ITERATIONS = int(5e4)

#     # Initialize components of function:
#     X0 = (
#         X[Y == np.min(Y), :].ravel().to(torch.float64)
#         if init_X0 is None
#         else torch.Tensor(init_X0).to(torch.float64)
#     )
#     V = (
#         torch.rand(dim, dim).to(torch.float64)
#         if init_V is None
#         else torch.Tensor(init_V).to(torch.float64)
#     )
#     C = (
#         torch.Tensor([np.min(Y)]).to(torch.float64)
#         if init_C is None
#         else torch.Tensor(init_C).to(torch.float64)
#     )

#     V.requires_grad = True
#     X0.requires_grad = True
#     C.requires_grad = True

#     # Initializing optimizer
#     if optim_newton:
#         optimizer = torch.optim.Adam([V, X0, C], lr=LEARNING_RATE)    
#     else:
        
#     optimizer = torch.optim.LBFGS([V, X0, C])
    
#     loss = 0
#     # Fit parameters to data:
#     for _ in range(ITERATIONS):
#         # Decrease learning rate over time:
#         if _ % 1000 == 0:
#             LEARNING_RATE = max(LEARNING_RATE / 10, float(1e-5))
#             print(loss)
#         optimizer.zero_grad()
#         loss = 0
#         for x, y in zip(X, Y):
#             p = (
#                 (x - X0) @ V @ V.T @ (x.T - X0.T)
#                 + torch.abs(C)
#                 + torch.Tensor([minimum])
#             )
#             loss = loss + torch.abs(p - y)
#         loss.backward()
#         optimizer.step()

#     # Convert types to numpy for output:
#     X0 = X0.detach().numpy()
#     V = V.detach().numpy()
#     C = torch.abs(C).detach().numpy() + minimum
#     print(loss)
#     return X0, V, C


# def fit_quadratic(X, Y):
#     dim = X.shape[1]
#     X = torch.tensor(X)
#     #  xAx + bx + c
#     V = torch.rand(dim, dim).to(torch.float64)
#     b = torch.rand(1, dim).to(torch.float64)
#     c = torch.rand(1).to(torch.float64)

#     V.requires_grad = True
#     b.requires_grad = True
#     c.requires_grad = True

#     optimizer = torch.optim.Adam([V, b, c], lr=float(1e-2))

#     for _ in range(1000):
#         optimizer.zero_grad()
#         loss = 0
#         for x, y in zip(X, Y):
#             p = x @ V @ V.T @ x.T + b @ x.T + c
#             loss = loss + torch.abs(p - y)
#         loss.backward()
#         optimizer.step()

#     A = (V @ V.T).detach().numpy()
#     b = b.detach().numpy()
#     c = c.detach().numpy()

#     return A, b, c
