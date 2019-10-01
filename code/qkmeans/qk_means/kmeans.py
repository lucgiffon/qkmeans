import copy

import numpy as np
from qkmeans.qk_means.utils import get_distances, compute_objective, assign_points_to_clusters, get_squared_froebenius_norm_line_wise, update_clusters_with_integrity_check
from qkmeans.utils import logger


def kmeans(X_data, K_nb_cluster, nb_iter, initialization,
           delta_objective_error_threshold=1e-6):

    X_data_norms = get_squared_froebenius_norm_line_wise(X_data)

    # plt.figure()
    # plt.yscale("log")

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

        objective_function[i_iter,] = compute_objective(X_data, U_centroids, indicator_vector)

        # Update centroid location using the newly
        # assigned data point classes
        # for c in range(K_nb_cluster):
        #     U_centroids_hat[c] = np.mean(X_data[indicator_vector == c], 0)

        cluster_names, counts = np.unique(indicator_vector, return_counts=True)
        cluster_names_sorted = np.argsort(cluster_names)

        counts, cluster_names_sorted = update_clusters_with_integrity_check(X_data,
                                                                            X_data_norms,
                                                                            U_centroids_hat,
                                                                            K_nb_cluster,
                                                                            counts,
                                                                            indicator_vector,
                                                                            distances,
                                                                            cluster_names,
                                                                            cluster_names_sorted)

        # check if all clusters still have points
        # for c in range(K_nb_cluster):
        #     biggest_cluster_index = np.argmax(counts)  # type: int
        #     biggest_cluster = cluster_names[biggest_cluster_index]
        #     biggest_cluster_data = X_data[indicator_vector == biggest_cluster]
        #
        #     cluster_data = X_data[indicator_vector == c]
        #     if len(cluster_data) == 0:
        #         logger.warning("cluster has lost data, add new cluster. cluster idx: {}".format(c))
        #         U_centroids_hat[c] = biggest_cluster_data[np.random.randint(len(biggest_cluster_data))].reshape(1, -1)
        #         counts = list(counts)
        #         counts[biggest_cluster_index] -= 1
        #         counts.append(1)
        #         counts = np.array(counts)
        #         cluster_names_sorted = list(cluster_names_sorted)
        #         cluster_names_sorted.append(c)
        #         cluster_names_sorted = np.array(cluster_names_sorted)
        #     else:
        #         U_centroids_hat[c] = np.mean(X_data[indicator_vector == c], 0)

        U_centroids = U_centroids_hat


        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1] # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée


        i_iter += 1

    return objective_function[:i_iter], U_centroids, indicator_vector