import matplotlib.pyplot as plt
import time

import logging
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import copy

import numpy as np
from pyqalm.qk_means.utils import get_distances, compute_objective, assign_points_to_clusters, get_squared_froebenius_norm_line_wise, update_clusters, \
    get_squared_froebenius_norm_line_wise_batch_by_batch
from pyqalm.utils import logger, DataGenerator
from sklearn import datasets


def kmeans_minibatch(X_data, K_nb_cluster, nb_iter, initialization, batch_size):

    logger.debug("Compute squared froebenius norm of data")
    X_data_norms = get_squared_froebenius_norm_line_wise_batch_by_batch(X_data, batch_size)

    # plt.figure()
    # plt.yscale("log")

    # Initialize our centroids by picking random data points

    U_centroids = copy.deepcopy(initialization)
    objective_function = np.empty((nb_iter,))

    total_nb_of_minibatch = X_data.shape[0] // batch_size

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while i_iter < nb_iter and (delta_objective_error > delta_objective_error_threshold):
        logger.info("Iteration number {}/{}".format(i_iter, nb_iter))
        full_count_vector = np.zeros(K_nb_cluster, dtype=int)
        full_indicator_vector = np.zeros(X_data.shape[0], dtype=int)
        U_centroids_before = np.copy(U_centroids)
        objective_value_so_far  = 0
        U_centroids = np.zeros_like(U_centroids_before)
        for i_minibatch, example_batch_indexes in enumerate(DataGenerator(X_data, batch_size=batch_size, return_indexes=True)):
            logger.info("Minibatch number {}/{}; Iteration number {}/{}".format(i_minibatch, total_nb_of_minibatch, i_iter, nb_iter))
            example_batch = X_data[example_batch_indexes]
            example_batch_norms = X_data_norms[example_batch_indexes]




            indicator_vector, distances = assign_points_to_clusters(example_batch, U_centroids_before, X_norms=example_batch_norms)
            full_indicator_vector[example_batch_indexes] = indicator_vector


            cluster_names, counts = np.unique(indicator_vector, return_counts=True)
            count_vector = np.zeros(nb_clust)
            count_vector[cluster_names] = counts

            full_count_vector = update_clusters(example_batch,
                                                U_centroids,
                                                K_nb_cluster,
                                                full_count_vector,
                                                count_vector,
                                                indicator_vector)


            # Update centroid location using the newly
            # assigned data point classes

            objective_value_so_far += np.sqrt(compute_objective(example_batch, U_centroids, indicator_vector))

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



        objective_function[i_iter,] = objective_value_so_far ** 2

        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1] # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée

        i_iter += 1

    return objective_function[:i_iter], U_centroids, indicator_vector

if __name__ == "__main__":
    batch_size = 10000
    nb_clust = 1000
    nb_iter = 30

    X = np.memmap("/home/luc/PycharmProjects/qalm_qmeans/data/external/blobs_1_billion.dat", mode="r", dtype="float32", shape=(int(1e6), 2000))

    logger.debug("Initializing clusters")
    centroids_init = X[np.random.permutation(X.shape[0])[:nb_clust]]

    start = time.time()
    logger.debug("Nb iteration: {}".format(nb_iter))
    obj, _, _ = kmeans_minibatch(X, nb_clust, nb_iter, centroids_init, batch_size)
    stop = time.time()
    plt.plot(obj)
    plt.show()
    print("It took {} s".format(stop - start))