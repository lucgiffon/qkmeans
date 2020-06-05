"""
simple Kmeans implementation.
"""
import copy

import numpy as np
from qkmeans.core.utils import compute_objective, assign_points_to_clusters, get_squared_froebenius_norm_line_wise, update_clusters_with_integrity_check
from qkmeans.utils import logger


def kmeans(X_data, K_nb_cluster, nb_iter, initialization,
           delta_objective_error_threshold=1e-6):
    """

    :param X_data: The data matrix of n examples in dimensions d in shape (n, d).
    :param K_nb_cluster: The number of clusters to look for.
    :param nb_iter: The maximum number of iteration.
    :param initialization: The (K, d) matrix of centroids at initialization.
    :param delta_objective_error_threshold: The normalized difference between the error criterion at 2 successive step must be greater or equal to that value.
    :return:
    """

    X_data_norms = get_squared_froebenius_norm_line_wise(X_data)


    # Initialize our centroids by picking random data points
    U_centroids_hat = copy.deepcopy(initialization)
    U_centroids = U_centroids_hat

    objective_function = np.empty((nb_iter,))

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error = np.inf
    while (i_iter == 0) or ((i_iter < nb_iter) and (delta_objective_error > delta_objective_error_threshold)):

        logger.info("Iteration Kmeans {}".format(i_iter))

        indicator_vector, distances = assign_points_to_clusters(X_data, U_centroids, X_norms=X_data_norms)



        cluster_names, counts = np.unique(indicator_vector, return_counts=True)
        cluster_names_sorted = np.argsort(cluster_names)

        # Update centroid location using the new indicator vector
        counts, cluster_names_sorted = update_clusters_with_integrity_check(X_data,
                                                                            X_data_norms,
                                                                            U_centroids_hat,
                                                                            K_nb_cluster,
                                                                            counts,
                                                                            indicator_vector,
                                                                            distances,
                                                                            cluster_names,
                                                                            cluster_names_sorted)

        U_centroids = U_centroids_hat

        objective_function[i_iter,] = compute_objective(X_data, U_centroids, indicator_vector)


        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1]


        i_iter += 1

    return objective_function[:i_iter], U_centroids, indicator_vector