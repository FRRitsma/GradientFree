# %%
import matplotlib.pyplot as plt
from functions import rosenbrock_np, fit_quadratic, fit_quadratic_2
import numpy as np
from sklearn.decomposition import PCA

# %%
n = 6
X = 4 * np.random.rand(1, 2) - np.array([2, 1])
X = X + float(1e-3) * np.random.rand(n, 2)
Y = rosenbrock_np(X)

gXX = np.empty([0, 0])
gYY = np.empty([0, 0])
gZZ = np.empty([0])

# %%
print(Y)

fit_x0, fit_V, fit_B = fit_quadratic_2(X, Y)

X = np.vstack([X, fit_x0])
Y = np.hstack([Y, rosenbrock_np(fit_x0)])

# %%
print(Y)

fit_x0, fit_V, fit_B = fit_quadratic_2(
    X[-8:,:],
    Y[-8:],
    init_X0=fit_x0,
    init_V=fit_V,
    init_C=fit_B,
)

X = np.vstack([X, fit_x0])
Y = np.hstack([Y, rosenbrock_np(fit_x0)])


# %% Compare fit to original:
x = X[1, :]

y = (x - fit_x0) @ (fit_V @ fit_V.T) @ (x.T - fit_x0.T) + fit_B


# %% TODO Visualization of rosenbrock function

# Make grid:
def grid_function_visualize(X: np.ndarray):
    # Edges:
    xmin = np.min(X, axis=0)
    xmax = np.max(X, axis=0)

    # Grid density:
    dens = max((xmax[0] - xmin[0]) / 20, (xmax[1] - xmin[1]) / 20)

    # Grid edge points:
    xx = np.arange(xmin[0], xmax[0], dens)
    yy = np.arange(xmin[1], xmax[1], dens)

    # Grid points:
    gxx, gyy = np.meshgrid(xx, yy)
    gxx, gyy = gxx.reshape(-1, 1), gyy.reshape(-1, 1)

    return gxx, gyy


gxx, gyy = grid_function_visualize(X[-4:, :])
gzz = rosenbrock_np(np.hstack([gxx, gyy]))

gXX = np.vstack([gXX.reshape(-1, 1), gxx])
gYY = np.vstack([gYY.reshape(-1, 1), gyy])


gZZ = np.hstack([gZZ, gzz])

plt.scatter(gXX, gYY, c=gZZ, s=500, marker="s")
plt.scatter(X[:, 0], X[:, 1], c="black")


xiter = new_iteration_point(X, Y, n_components=2)
# xiter += np.random.rand(2)*float(1e-3)
yiter = rosenbrock_np(xiter)

# Append
X = np.vstack([X, xiter])
Y = np.hstack([Y, yiter])


# %%
plt.scatter(X[:, 0], X[:, 1], c=Y)

# %%


def new_iteration_point(X, Y, n_components: int = None):
    if n_components is None:
        n_components = X.shape[1]

    assert n_components <= X.shape[1]

    # Combine X, Y:
    P = np.hstack([X, Y.reshape(-1, 1)])

    # Fit principal components analysis:
    pca = PCA(n_components=1)
    pca.fit(P)

    # Get low dimensional representation:
    sub_X = pca.transform(P)

    # Fit quadratic function:
    A, B, _ = fit_quadratic(sub_X, Y)

    # Lowest point in low dimensional representation:
    sub_x0 = -0.5 * (np.linalg.inv(A) @ B.T).T

    # Iteration point:
    iter_X = pca.inverse_transform(sub_x0)

    return iter_X[:, 0:2]
