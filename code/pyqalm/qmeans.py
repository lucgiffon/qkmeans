"""
kmeans algorithm inspired from https://jonchar.net/notebooks/k-means/ .
"""
import logging
import daiquiri

import numpy as np
from pyqalm.qalm import HierarchicalPALM4MSA
from sklearn import datasets
import matplotlib.pyplot as plt

from pyqalm.utils import get_side_prod, logger, get_lambda_proxsplincol, constant_proj

daiquiri.setup(level=logging.INFO)

def initialize_clusters(K_nb_clusters, d_dimension, nb_factors):
    """
    Initialize clusters as a product of sparse matrices.

    :return: The list of the sparse matrices whose product gives the initial clusters
    """
    min_K_d = min([K_nb_clusters, d_dimension])

    lst_factors = [np.eye(min_K_d) for i in range(nb_factors) ]
    lst_factors[-1] = np.random.rand(min_K_d, d_dimension)
    lst_factors[0] = np.zeros((K_nb_clusters, min_K_d))
    lst_factors[0][np.diag_indices(min_K_d)] = np.random.rand(min_K_d)
    # dansle papier gribonval le facteur de droite est initialisé "full zero"
    # mais ça marche aussi avec n'importe quelle valeur...ce qui serait nécessaire dans notre cas
    # todo pourquoi voulaient-ils mettre des zéros?

    U_centroids = get_side_prod(lst_factors)
    return lst_factors, U_centroids

def get_distances(X_data, U_centroids):
    """
    Return the matrice of distance between each data point and each centroid.
    :param X_data:
    :param U_centroids:
    :return:
    """
    centroid_norms = np.linalg.norm(U_centroids, axis=1)
    # todo tirer parti de la sparsité des matrices.
    centroid_distances = 2*(U_centroids @ X_data.T) + centroid_norms[:, np.newaxis]

    return centroid_distances.T



def qmeans(X_data, K_nb_cluster, nb_iter, params_palm4msa):
    def compute_objective(X_data, U_centroids, indicator_vector):
        return np.linalg.norm(X_data - U_centroids[indicator_vector])

    plt.figure()
    # plt.yscale("log")

    nb_factors = params_palm4msa["nb_factors"]
    init_lambda = params_palm4msa["init_lambda"]
    nb_iter_palm = params_palm4msa["nb_iter"]

    # Initialize our centroids by picking random data points
    U_centroids_hat = X_data[np.random.permutation(X_data.shape[0])[:K_nb_cluster]]
    # lst_factors, U_centroids = initialize_clusters(K_nb_cluster, X_data.shape[1], nb_factors)
    min_K_d = min(U_centroids_hat.shape)
    U_centroids = U_centroids_hat

    lst_factors = [np.eye(min_K_d) for _ in range(nb_factors)]
    lst_factors[0] = np.eye(U_centroids_hat.shape[0], min_K_d)
    lst_factors[-1] = np.zeros((min_K_d, U_centroids_hat.shape[1]))

    lst_proj_op_by_fac_step = []
    factor = 4
    nb_keep_values = factor * min_K_d

    for k in range(nb_factors - 1):
        nb_values_residual = max(nb_keep_values, int(min_K_d / 2 ** (k + 1)) * min_K_d)
        if k == 0:
            dct_step_lst_nb_keep_values = {
                "split": [constant_proj] * 2,
                "finetune": [constant_proj] * ((k + 1) + 1)
            }
        else:
            dct_step_lst_nb_keep_values = {
                "split": [get_lambda_proxsplincol(nb_keep_values), get_lambda_proxsplincol(nb_values_residual)],
                "finetune": [get_lambda_proxsplincol(nb_keep_values)] * (k + 1) + [get_lambda_proxsplincol(nb_values_residual)]
            }
        lst_proj_op_by_fac_step.append(dct_step_lst_nb_keep_values)

    objective_function = np.empty((nb_iter,))

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error = 1e-6
    first_iter = True
    while ((i_iter < nb_iter) and (objective_function[i_iter - 1] > delta_objective_error)) or (first_iter):
        logger.info("Iteration Qmeans {}".format(i_iter))

        distances = get_distances(X_data, U_centroids)

        # then, Determine class membership of each point
        # by picking the closest centroid
        indicator_vector = np.argmin(distances, axis=1)

        cluster_names, counts = np.unique(indicator_vector, return_counts=True)
        cluster_names_sorted = np.argsort(cluster_names)

        diag_counts = np.diag(np.sqrt(counts[cluster_names_sorted])) # todo use sparse matrix object

        objective_function[i_iter] = compute_objective(X_data, U_centroids, indicator_vector)
        lst_factors[0] = diag_counts

        # Assign all points to the nearest centroid
        # first get distance from all points to all centroids
        _lambda, lst_factors, U_centroids, nb_iter_by_factor = HierarchicalPALM4MSA(
            arr_X_target=diag_counts @ U_centroids_hat,
            lst_S_init=lst_factors,
            lst_dct_projection_function=lst_proj_op_by_fac_step,
            f_lambda_init=init_lambda,
            nb_iter=nb_iter_palm,
            update_right_to_left=True,
            residual_on_right=True,
            graphical_display=False)

        # Update centroid location using the newly
        # assigned data point classes
        for c in range(K_nb_cluster):
            U_centroids_hat[c] = np.mean(X_data[indicator_vector == c], 0)

        if np.isnan(U_centroids_hat).any():
            exit("Some clusters have no point. Aborting iteration {}".format(i_iter))

        plt.scatter(i_iter, objective_function[i_iter])
        plt.pause(1)

        i_iter += 1

    plt.figure()
    plt.semilogy(objective_function[:i_iter])
    plt.pause(0.01)

if __name__ == '__main__':
    X, _ = datasets.make_blobs(n_samples=1000, n_features=200, centers=20)
    qmeans(X, 5, 100, {"nb_factors": 4, "init_lambda": 1., "nb_iter": 300})
