# %%
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

dim = 2
sigma = np.array([[4, 1], [1, 2]])
mu = np.array([3, 3])


X = np.random.multivariate_normal(mu, sigma, 1000)

pca = PCA(n_components=1)
pca.fit(X)
reco = pca.inverse_transform(pca.transform(X))

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(reco[:, 0], reco[:, 1])
plt.show()
