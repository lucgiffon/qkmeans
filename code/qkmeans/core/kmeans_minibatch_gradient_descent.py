import matplotlib.pyplot as plt

import logging
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import copy

import numpy as np
from qkmeans.core.utils import get_distances, compute_objective, assign_points_to_clusters, get_squared_froebenius_norm_line_wise, update_clusters_with_integrity_check
from qkmeans.utils import logger, DataGenerator
from sklearn import datasets


def kmeans_minibatch(X_data, K_nb_cluster, nb_iter, initialization, batch_size):

    X_data_norms = get_squared_froebenius_norm_line_wise(X_data)

    # plt.figure()
    # plt.yscale("log")

    # Initialize our centroids by picking random data points
    U_centroids_hat = copy.deepcopy(initialization)
    U_centroids = U_centroids_hat
    full_indicator_vector = np.zeros(X_data.shape[0], dtype=int)
    full_count_vector = np.zeros(K_nb_cluster, dtype=int)
    objective_function = np.empty((nb_iter,))

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while True:
        for i_iter, example_batch_indexes in enumerate(DataGenerator(X_data, batch_size=batch_size, return_indexes=True)):
            if not (delta_objective_error > delta_objective_error_threshold):
                logger.info("not (delta_objective_error {}-{}={} > delta_objective_error_threshold {})".format(objective_function[i_iter], objective_function[i_iter-1], delta_objective_error, delta_objective_error_threshold))
                break

            example_batch = X_data[example_batch_indexes]

            logger.info("Iteration Kmeans {}".format(i_iter))

            indicator_vector, distances = assign_points_to_clusters(example_batch, U_centroids, X_norms=X_data_norms[example_batch_indexes])
            full_indicator_vector[example_batch_indexes] = indicator_vector


            cluster_names, counts = np.unique(indicator_vector, return_counts=True)
            # cluster_names_sorted = np.argsort(cluster_names)
            #
            count_vector = np.zeros(K_nb_cluster, dtype=int)
            count_vector[cluster_names] = counts

            full_count_vector += count_vector
            # previous_full_count_vector = full_count_vector - count_vector

            # Update centroid location using the newly
            # assigned data point classes
            # This way of updating the centroids (centroid index wise) is better than the one proposed in the paper "Web-Scale K-Means Clustering"
            # as the number of update with always be <= batch_size
            for c in range(K_nb_cluster):
                if full_count_vector[c] != 0 and count_vector[c] != 0:
                    U_centroids_hat[c] += (1/full_count_vector[c]) * np.sum(example_batch[indicator_vector == c] - U_centroids_hat[c], axis=0)
                    # this is exactly equivalent to an update of the mean:
                    # U_centroids_hat[c] = (previous_full_count_vector[c] / full_count_vector[c]) * U_centroids_hat[c] + (1 / full_count_vector[c]) * np.sum(example_batch[indicator_vector == c], axis=0)


            # for i_ex, ex in enumerate(example_batch):
            #     c = indicator_vector[i_ex]
            #     full_count_vector[c] += 1
            #     eta = 1./full_count_vector[c]
            #     U_centroids_hat[c] = (1-eta) * U_centroids_hat[c] + eta * ex

            # counts, cluster_names_sorted = assess_clusters_integrity(X_data,
            #                                                          X_data_norms,
            #                                                          U_centroids_hat,
            #                                                          K_nb_cluster,
            #                                                          counts,
            #                                                          indicator_vector,
            #                                                          distances,
            #                                                          cluster_names,
            #                                                          cluster_names_sorted)

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

            objective_function[i_iter,] = compute_objective(X_data, U_centroids, full_indicator_vector)

            if i_iter >= 1:
                delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1] # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée

            i_iter += 1
        else:
            continue
        break

    return objective_function[:i_iter], U_centroids, indicator_vector

if __name__ == "__main__":
    n_samples = 1000
    n_features = 2
    n_centers = 500

    batch_size = 100
    nb_clust = 10
    nb_iter = 20

    X, _ = datasets.make_blobs(n_samples=n_samples,
                               n_features=n_features,
                               centers=n_centers)

    centroids_init = X[np.random.permutation(X.shape[0])[:nb_clust]]

    actual_nb_iter = (n_samples // batch_size) * nb_iter

    logger.info("Nb iteration: {}".format(actual_nb_iter))
    obj, _, _ = kmeans_minibatch(X, nb_clust, actual_nb_iter, centroids_init, batch_size)

    plt.plot(obj)
    plt.show()