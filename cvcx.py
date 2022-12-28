# %%
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from functions import parameter_count, triangle

from itertools import combinations_with_replacement  # ('ABCD', 2)


A = np.array([[1, 19039], [1, 47792], [1, 9672], [1, 32521], [1, 11409], [1, 58843]])
b = np.array(
    [[0.05938044], [0.27213514], [0.00252875], [0.18535543], [0.01959069], [0.52605937]]
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

# plt.figure()
# plt.scatter(A[:, 1], b)
# plt.plot(A[:, 1], np.multiply(A[:, 1], slope) + intercept)

# Order of constants:

# Toy problem:
# a*x**2 + b*x + c, a=1, b=2, c=3

x = np.arange(4)
y = x**2 + 2 * x + 3

x = x.reshape([-1, 1])
var_vector = np.empty([x.shape[0], 3])


def fill_row_with_variables(x):
    dim = len(x)
    a = np.empty([parameter_count(dim)])
    for i, (x0, x1) in enumerate(combinations_with_replacement(x, 2)):
        a[i] = x0 * x1
    for i, x0 in enumerate(x):
        a[i + triangle(dim)] = x0
    a[-1] = 1
    print(a)

fill_row_with_variables([2,3])
