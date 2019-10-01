"""
Kmeans implementation with batch-by-batch read of the input dataset.

The algorithm convergence is strictly equivalent to the Kmeans algorithm.
"""

import matplotlib.pyplot as plt
import time

import logging
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import copy

import numpy as np
from qkmeans.core.utils import compute_objective, assign_points_to_clusters, update_clusters, \
    get_squared_froebenius_norm_line_wise_batch_by_batch
from qkmeans.utils import logger, DataGenerator


def kmeans_minibatch(X_data,
                     K_nb_cluster,
                     nb_iter,
                     initialization,
                     batch_size,
                     delta_objective_error_threshold=1e-6):
    """

    :param X_data: The data matrix of n examples in dimensions d in shape (n, d).
    :param K_nb_cluster: The number of clusters to look for.
    :param nb_iter: The maximum number of iteration.
    :param initialization: The (K, d) matrix of centroids at initialization.
    :param batch_size: The size of each batch.
    :param delta_objective_error_threshold: The normalized difference between the error criterion at 2 successive step must be greater or equal to that value.
    :return:
    """

    logger.debug("Compute squared froebenius norm of data")
    X_data_norms = get_squared_froebenius_norm_line_wise_batch_by_batch(X_data, batch_size)

    # Initialize our centroids by picking random data points

    U_centroids = copy.deepcopy(initialization)
    objective_function = np.empty((nb_iter,))

    total_nb_of_minibatch = X_data.shape[0] // batch_size

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error = np.inf
    while i_iter < nb_iter and (delta_objective_error > delta_objective_error_threshold):
        logger.info("Iteration number {}/{}".format(i_iter, nb_iter))
        # Prepare next epoch
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
            count_vector = np.zeros(K_nb_cluster)
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


        objective_function[i_iter,] = objective_value_so_far ** 2

        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1] # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée

        i_iter += 1

    return objective_function[:i_iter], U_centroids, full_indicator_vector

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