from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster._kmeans import _k_init
from sklearn.utils.extmath import row_norms
from sklearn.datasets import make_blobs

k = 4
seed = np.random.RandomState(0)
X, y = make_blobs(1000, centers=k)
x_squared_norms = row_norms(X, squared=True)
centers = _k_init(X, k, x_squared_norms, random_state=seed)


random_centers = X[seed.permutation(X.shape[0])[:k]]
plt.scatter(X[:, 0], X[:, 1], c="c")
plt.scatter(centers[:, 0], centers[:, 1], c="g", label="kmeans++")
plt.scatter(random_centers[:, 0], random_centers[:, 1], c="r", label="random uniform")
plt.legend()
plt.show()
