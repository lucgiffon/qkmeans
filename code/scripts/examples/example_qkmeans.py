import logging
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pformat

import daiquiri
import numpy as np
from qkmeans.core.kmeans import kmeans
from qkmeans.core.qmeans_fast import qmeans
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.utils import logger
from sklearn import datasets

np.random.seed(0)
daiquiri.setup(level=logging.INFO)


# Create dataset
n_samples = 1000
n_features = 20
n_centers = 50
X, _ = datasets.make_blobs(n_samples=n_samples,
                           n_features=n_features,
                           centers=n_centers)

# Initialize clustering
nb_clusters = 10
nb_iter_kmeans = 10
nb_factors = 5
U_centroids_hat = X[np.random.permutation(X.shape[0])[:nb_clusters]]
# kmeans++ initialization is not feasible because complexity is O(ndk)...

# Initialize palm4msa
sparsity_factor = 2
nb_iter_palm = 30
delta_objective_error_threshold_in_palm = 1e-6
# Create constraints for palm4msa
lst_constraints, lst_constraints_vals = build_constraint_set_smart(
    U_centroids_hat.shape[0], U_centroids_hat.shape[1], nb_factors,
    sparsity_factor=sparsity_factor, residual_on_right=True)

logger.info("Description of constraints: \n{}".format(pformat(lst_constraints_vals)))

hierarchical_palm_init = {
    "init_lambda": 1.,
    "nb_iter": nb_iter_palm,
    "lst_constraint_sets": lst_constraints,
    "residual_on_right": True,
    "delta_objective_error_threshold": delta_objective_error_threshold_in_palm,
    "track_objective": False
}

logger.info('Running QuicK-means with H-Palm')

# QKmeans with hierarchical palm4msa
objective_function_with_hier_palm, op_centroids_hier, indicator_hier, lst_objective_function_hier_palm = \
    qmeans(X,
           nb_clusters,
           nb_iter_kmeans,
           nb_factors,
           hierarchical_palm_init,
           initialization=U_centroids_hat,
           hierarchical_inside=True)


# QKmeans with simple palm4msa
logger.info('Running QuicK-means with Palm')
objective_function_with_palm, op_centroids_palm, indicator_palm, lst_objective_function_palm = \
    qmeans(X, nb_clusters, nb_iter_kmeans, nb_factors,
           hierarchical_palm_init,
           initialization=U_centroids_hat)


# Kmeans with lloyd algorithm
logger.info('Running K-means')
objective_values_k, centroids_finaux, indicator_kmean = \
    kmeans(X, nb_clusters, nb_iter_kmeans,
           initialization=U_centroids_hat)

logger.info('Display')
plt.figure()

plt.plot(np.arange(len(objective_function_with_hier_palm)), objective_function_with_hier_palm, marker="x", label="hierarchical")
plt.plot(np.arange(len(objective_function_with_palm)), objective_function_with_palm, marker="x", label="palm")
plt.plot(np.arange(len(objective_values_k)), objective_values_k, marker="x", label="kmeans")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
