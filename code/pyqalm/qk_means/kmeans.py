import copy

import numpy as np
from pyqalm.qk_means.utils import get_distances, compute_objective
from pyqalm.utils import logger


def kmeans(X_data, K_nb_cluster, nb_iter, initialization):

    # plt.figure()
    # plt.yscale("log")

    # Initialize our centroids by picking random data points
    U_centroids_hat = copy.deepcopy(initialization)
    U_centroids = U_centroids_hat

    objective_function = np.empty((nb_iter,))

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while (i_iter == 0) or ((i_iter < nb_iter) and (delta_objective_error > delta_objective_error_threshold)):

        logger.info("Iteration Kmeans {}".format(i_iter))

        # Assign all points to the nearest centroid
        # first get distance from all points to all centroids
        distances = get_distances(X_data, U_centroids)
        # then, Determine class membership of each point
        # by picking the closest centroid
        indicator_vector = np.argmin(distances, axis=1)

        objective_function[i_iter,] = compute_objective(X_data, U_centroids, indicator_vector)

        # Update centroid location using the newly
        # assigned data point classes
        for c in range(K_nb_cluster):
            U_centroids_hat[c] = np.mean(X_data[indicator_vector == c], 0)

        U_centroids = U_centroids_hat

        if np.isnan(U_centroids_hat).any():
            exit("Some clusters have no point. Aborting iteration {}".format(i_iter))

        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1] # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée


        i_iter += 1

    return objective_function[:i_iter], U_centroids, indicator_vector