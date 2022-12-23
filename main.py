# %%
import numpy as np
import plotly.express as px
import scipy

spdist = scipy.spatial.distance.cdist


"""
Domain of function: 
x [-2, 2]
y [-1, 3]

Minimum at:
[1, 1]
"""

# %%


def iter_diff(array: np.ndarray, iter: int = 1) -> np.ndarray:
    iter += 1
    shape = array.shape
    array = array.reshape(shape[0], -1)
    return_array = np.empty([0, array.shape[1]])
    for i in range(1, iter):
        diff = array[i:, :] - array[:-i, :]
        return_array = np.vstack([return_array, diff.reshape(diff.shape[0], -1)])
    if len(shape) == 1:
        return_array = return_array.reshape(-1)
    return return_array


def rosenbrock_np(xy: np.ndarray) -> np.ndarray:
    xy = xy.reshape(-1, 2)
    return rosenbrock(xy[:, 0], xy[:, 1])


def rosenbrock(x: float, y: float) -> float:
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x**2) ** 2


class StoreResults:
    X = np.empty([0, 2])
    Y = np.empty([0])
    step_size = float(1e-4)

    def process(self, x):
        y = rosenbrock_np(x)
        # Append:
        self.X = np.vstack([self.X, x.reshape(1, 2)])
        self.Y = np.hstack([self.Y, y])

    @staticmethod
    def __noise_on_step(step: np.ndarray):
        # Random direction:
        random = 1 / 2 + np.random.rand()
        random = random / np.linalg.norm(random)
        # Scale of input step:
        scale = np.linalg.norm(step)
        return step + float(1e-1) * scale * random

    def get_next(self):
        if len(self.X) == 0:
            return 4 * np.random.rand(2) - np.array([2, 1])
        if len(self.X) == 1:
            step = np.random.rand(2)
            step = step / sum(abs(step))
            return self.X[0, :] + self.step_size * step
        # Get lowest point:
        idx = np.argmin(self.Y)
        argsort = np.argsort(spdist(self.X[idx, :].reshape(1, 2), self.X))
        argsort = argsort.ravel()
        idx1 = argsort[1]
        # Get derivative:
        derv = (self.Y[idx] - self.Y[idx1]) / (self.X[idx, :] - self.X[idx1, :])
        derv = derv / abs(sum(derv))
        return self.X[idx, :] - derv * self.step_size



n = 30
X = 4 * np.random.rand(1, 2) - np.array([2, 1])
X = X + float(1e-3) * np.random.rand(n, 2)
Y = rosenbrock_np(X)

from sklearn.decomposition import PCA
df = iter_diff(Y, n-1).reshape(-1, 1) / iter_diff(X, n-1)
pca = PCA(n_components=1)
pca.fit(df)
plt.scatter(pca.transform(X), Y)
plt.show()

# %%
