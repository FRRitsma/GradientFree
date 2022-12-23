# %%
import torch
import numpy as np


# Create function:
V = np.random.rand(2, 2)
A = V.T @ V
X = -1 / 2 + 2 * np.random.rand(10, 2)


def quadratic_func(A, b, c, X):
    Y = np.empty(0)
    for x in X:
        y = x @ A @ x.T + b @ x.T + c
        print(y)
        Y = np.hstack([Y, y])
    return Y


def quadratic_minimum(A, b):
    return (-0.5 * np.linalg.inv(A) @ b.T).T


V = np.random.rand(3, 3)
A = V.T @ V
A = np.eye(3)
b = np.random.rand(1, 3)
b = np.ones([1, 3])
c = np.random.rand(1)

X = np.random.rand(20, 3)
Y = quadratic_func(A, b, c, X)

x0 = -0.5 * (np.linalg.inv(A) @ b.T).T
y0 = quadratic_func(A, b, c, x0)

#%%
At, bt, ct = fit_quadratic(X, Y)


# %%
Vt = torch.rand(2, 2)
Vt.requires_grad = True
x0 = torch.Tensor([1, 0])
x1 = torch.Tensor([0, 1])

optimizer = torch.optim.Adam([Vt], lr=float(1e-2))
for i in range(1000):
    optimizer.zero_grad()
    loss = x0 @ Vt.T @ Vt @ x0.T
    loss = loss + torch.abs(x1 @ Vt.T @ Vt @ x1.T - 1)
    print(loss)
    loss.backward()
    optimizer.step()

# %%


def fit_quadratic(X, Y):
    dim = X.shape[1]
    X = torch.tensor(X)
    #  xAx + bx + c
    V = torch.rand(dim, dim).to(torch.float64)
    b = torch.rand(1, dim).to(torch.float64)
    c = torch.rand(1).to(torch.float64)

    V.requires_grad = True
    b.requires_grad = True
    c.requires_grad = True

    optimizer = torch.optim.Adam([V, b, c], lr=float(1e-2))

    for _ in range(1000):
        optimizer.zero_grad()
        loss = 0
        for x, y in zip(X, Y):
            p = x @ V @ V.T @ x.T + b @ x.T + c
            loss = loss + torch.abs(p - y)
        loss.backward()
        optimizer.step()

    A = (V @ V.T).detach().numpy()
    b = b.detach().numpy()
    c = c.detach().numpy()

    return A, b, c


def new_fit_quadratic(X: np.ndarray, Y: np.ndarray, minimum: float):

    dim = X.shape[1]
    X = torch.tensor(X)
    V = torch.rand(dim, dim).to(torch.float64)
    x0 = torch.rand(1, dim).to(torch.float64)
    c = torch.rand(1).to(torch.float64)

    V.requires_grad = True
    x0.requires_grad = True
    c.requires_grad = True

    optimizer = torch.optim.Adam([V, x0, c], lr=float(1e-2))

    for _ in range(1000):
        optimizer.zero_grad()
        loss = 0
        for x, y in zip(X, Y):
            p = (x - x0) @ V @ V.T @ (x - x0).T + torch.abs(c) + minimum
            loss = loss + torch.abs(p - y)
        loss.backward()
        optimizer.step()

    A = (V @ V.T).detach().numpy()
    x0 = x0.detach().numpy()
    c = torch.abs(c).detach().numpy() + minimum

    return A, x0, c


At, x0, ct = new_fit_quadratic(X, Y, 0)


# An = At.detach().numpy()
# bn = bt.detach().numpy()
