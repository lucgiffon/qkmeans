"""
Accelerated version of the qkmeans_naive algorithm

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon

"""
import logging
import daiquiri
import copy
from collections import OrderedDict
from pprint import pformat

import numpy as np
from qkmeans.core.utils import compute_objective, assign_points_to_clusters, build_constraint_set_smart, get_squared_froebenius_norm_line_wise, update_clusters_with_integrity_check
from qkmeans.core.kmeans import kmeans
from scipy.sparse import csr_matrix
from sklearn import datasets
import matplotlib.pyplot as plt

from qkmeans.palm.palm_fast import hierarchical_palm4msa, \
    palm4msa
from qkmeans.data_structures import SparseFactors
from qkmeans.utils import logger


def init_lst_factors(d_in, d_out, p_nb_factors):
    """
    Return a simple initialization of a list of sparse factors in which all the factors are identity but the last one is zeros.

    :param d_in: left dimension
    :param d_out: right dimension
    :param p_nb_factors: number of factors
    :return:
    """
    min_K_d = min(d_in, d_out)

    lst_factors = [np.eye(min_K_d) for _ in range(p_nb_factors)]

    eye_norm = np.sqrt(d_in)
    lst_factors[0] = np.eye(d_in) / eye_norm
    lst_factors[1] = np.eye(d_in, min_K_d)
    lst_factors[-1] = np.zeros((min_K_d, d_out))

    return lst_factors

def qmeans(X_data: np.ndarray,
           K_nb_cluster: int,
           nb_iter: int,
           nb_factors: int,
           params_palm4msa: dict,
           initialization: np.ndarray,
           hierarchical_inside=False,
           delta_objective_error_threshold=1e-6,
           hierarchical_init=False):
    """

    :param X_data: The data matrix of n examples in dimensions d in shape (n, d).
    :param K_nb_cluster: The number of clusters to look for.
    :param nb_iter: The maximum number of iteration.
    :param nb_factors: The number of factors for the decomposition.
    :param initialization: The initial matrix of centroids not yet factorized.
    :param params_palm4msa: The dictionnary of parameters for the palm4msa algorithm.
    :param hierarchical_inside: Tell the algorithm if the hierarchical version of palm4msa should be used.
    :param delta_objective_error_threshold:
    :param hierarchical_init: Tells if the algorithm should make the initialization of sparse factors with the hierarchical version of palm or not.
    :return:
    """
    assert K_nb_cluster == initialization.shape[0], "The number of cluster {} is not equal to the number of centroids in the initialization {}.".format(K_nb_cluster, initialization.shape[0])

    X_data_norms = get_squared_froebenius_norm_line_wise(X_data)

    nb_examples = X_data.shape[0]

    logger.info("Initializing Qmeans")

    init_lambda = params_palm4msa["init_lambda"]
    nb_iter_palm = params_palm4msa["nb_iter"]
    lst_proj_op_by_fac_step = params_palm4msa["lst_constraint_sets"]
    residual_on_right = params_palm4msa["residual_on_right"]
    delta_objective_error_threshold_inner_palm = params_palm4msa["delta_objective_error_threshold"]
    track_objective_palm = params_palm4msa["track_objective"]

    X_centroids_hat = copy.deepcopy(initialization)

    lst_factors = init_lst_factors(K_nb_cluster, X_centroids_hat.shape[1], nb_factors)

    eye_norm = np.sqrt(K_nb_cluster)

    if hierarchical_inside or hierarchical_init:
        _lambda_tmp, op_factors, U_centroids, objective_palm, array_objective_hierarchical= \
            hierarchical_palm4msa(
                arr_X_target=np.eye(K_nb_cluster) @ X_centroids_hat,
                lst_S_init=lst_factors,
                lst_dct_projection_function=lst_proj_op_by_fac_step,
                f_lambda_init=init_lambda * eye_norm,
                nb_iter=nb_iter_palm,
                update_right_to_left=True,
                residual_on_right=residual_on_right,
                track_objective_palm=track_objective_palm,
                delta_objective_error_threshold_palm=delta_objective_error_threshold_inner_palm,
                return_objective_function=track_objective_palm)
    else:
        _lambda_tmp, op_factors, U_centroids, objective_palm, nb_iter_palm = \
            palm4msa(
                arr_X_target=np.eye(K_nb_cluster) @ X_centroids_hat,
                lst_S_init=lst_factors,
                nb_factors=len(lst_factors),
                lst_projection_functions=lst_proj_op_by_fac_step[-1][
                    "finetune"],
                f_lambda_init=init_lambda * eye_norm,
                nb_iter=nb_iter_palm,
                update_right_to_left=True,
                track_objective=track_objective_palm,
                delta_objective_error_threshold=delta_objective_error_threshold_inner_palm)

    lst_factors = None  # safe assignment for debug

    _lambda = _lambda_tmp / eye_norm

    objective_function = np.ones(nb_iter) * -1
    lst_all_objective_functions_palm = []
    lst_all_objective_functions_palm.append(objective_palm)

    i_iter = 0
    delta_objective_error = np.inf
    while ((i_iter < nb_iter) and (delta_objective_error > delta_objective_error_threshold)):

        logger.info("Iteration Qmeans {}".format(i_iter))

        lst_factors_ = op_factors.get_list_of_factors()
        op_centroids = SparseFactors([lst_factors_[1] * _lambda] + lst_factors_[2:])

        indicator_vector, distances = assign_points_to_clusters(X_data, op_centroids, X_norms=X_data_norms)

        objective_function[i_iter] = compute_objective(X_data, op_centroids, indicator_vector)

        # get the number of observation in each cluster
        cluster_names, counts = np.unique(indicator_vector, return_counts=True)
        cluster_names_sorted = np.argsort(cluster_names)

        # Update centroid location using the newly (it happens in the assess_cluster_integrity function)
        # assigned data point classes
        # and check if all clusters still have points
        # and change the object X_centroids_hat in place if some cluster have lost points (biggest cluster)
        counts, cluster_names_sorted = update_clusters_with_integrity_check(X_data,
                                                                            X_data_norms,
                                                                            X_centroids_hat,
                                                                            K_nb_cluster,
                                                                            counts,
                                                                            indicator_vector,
                                                                            distances,
                                                                            cluster_names,
                                                                            cluster_names_sorted)
        # create the diagonal of the sqrt of those counts
        diag_counts_sqrt_normalized = csr_matrix(
            (np.sqrt(counts[cluster_names_sorted] / nb_examples),
             (np.arange(K_nb_cluster), np.arange(K_nb_cluster))))
        diag_counts_sqrt = np.sqrt(counts[cluster_names_sorted])

        # set it as first factor
        op_factors.set_factor(0, diag_counts_sqrt_normalized)


        if hierarchical_inside:
            _lambda_tmp, op_factors, _, objective_palm, array_objective_hierarchical = \
                hierarchical_palm4msa(
                    arr_X_target=diag_counts_sqrt[:, None,] *  X_centroids_hat,
                    lst_S_init=op_factors.get_list_of_factors(),
                    lst_dct_projection_function=lst_proj_op_by_fac_step,
                    f_lambda_init=_lambda * np.sqrt(nb_examples),
                    nb_iter=nb_iter_palm,
                    update_right_to_left=True,
                    residual_on_right=residual_on_right,
                    return_objective_function=track_objective_palm,
                    track_objective_palm=track_objective_palm,
                    delta_objective_error_threshold_palm=delta_objective_error_threshold_inner_palm)

        else:
            _lambda_tmp, op_factors, _, objective_palm, nb_iter_palm = \
                palm4msa(arr_X_target=diag_counts_sqrt[:, None,] *  X_centroids_hat,
                         lst_S_init=op_factors.get_list_of_factors(),
                         nb_factors=op_factors.n_factors,
                         lst_projection_functions=lst_proj_op_by_fac_step[-1][
                             "finetune"],
                         f_lambda_init=_lambda * np.sqrt(nb_examples),
                         nb_iter=nb_iter_palm,
                         update_right_to_left=True,
                         track_objective=track_objective_palm,
                         delta_objective_error_threshold=delta_objective_error_threshold_inner_palm)

        lst_all_objective_functions_palm.append(objective_palm)

        _lambda = _lambda_tmp / np.sqrt(nb_examples)

        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1]

        # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilee

        i_iter += 1


    op_centroids = SparseFactors([lst_factors_[1] * _lambda] + lst_factors_[2:])

    return objective_function[:i_iter], op_centroids, indicator_vector, lst_all_objective_functions_palm


if __name__ == '__main__':
    np.random.seed(0)
    daiquiri.setup(level=logging.INFO)
    small_dim = True
    if small_dim:
        nb_clusters = 10
        nb_iter_kmeans = 10
        n_samples = 1000
        n_features = 20
        n_centers = 50
        nb_factors = 5
    else:
        nb_clusters = 1024
        nb_iter_kmeans = 10
        n_samples = 10000
        n_features = 64
        n_centers = 4096
        nb_factors = int(np.log2(min(nb_clusters, n_features)))
    X, _ = datasets.make_blobs(n_samples=n_samples,
                               n_features=n_features,
                               centers=n_centers)

    U_centroids_hat = X[np.random.permutation(X.shape[0])[:nb_clusters]]
    # kmeans++ initialization is not feasible because complexity is O(ndk)...
    residual_on_right = True

    sparsity_factor = 2
    nb_iter_palm = 30
    delta_objective_error_threshold_in_palm = 1e-6
    track_objective_in_palm = True

    lst_constraints, lst_constraints_vals = build_constraint_set_smart(
        U_centroids_hat.shape[0], U_centroids_hat.shape[1], nb_factors,
        sparsity_factor=sparsity_factor, residual_on_right=residual_on_right)
    logger.info("constraints: {}".format(pformat(lst_constraints_vals)))


    hierarchical_palm_init = {
        "init_lambda": 1.,
        "nb_iter": nb_iter_palm,
        "lst_constraint_sets": lst_constraints,
        "residual_on_right": residual_on_right,
        "delta_objective_error_threshold": delta_objective_error_threshold_in_palm,
        "track_objective": track_objective_in_palm
    }

    logger.info('Running QuicK-means with H-Palm')
    objective_function_with_hier_palm, op_centroids_hier, indicator_hier, lst_objective_function_hier_palm = \
        qmeans(X,
               nb_clusters,
               nb_iter_kmeans,
               nb_factors,
               hierarchical_palm_init,
               initialization=U_centroids_hat,
               hierarchical_inside=True)

    logger.info('Running QuicK-means with Palm')
    objective_function_with_palm, op_centroids_palm, indicator_palm, lst_objective_function_palm = \
        qmeans(X, nb_clusters, nb_iter_kmeans, nb_factors,
               hierarchical_palm_init,
               initialization=U_centroids_hat)

    try:
        logger.info('Running K-means')
        objective_values_k, centroids_finaux, indicator_kmean = \
            kmeans(X, nb_clusters, nb_iter_kmeans,
                   initialization=U_centroids_hat)
    except SystemExit as e:
        logger.info("There have been a problem in kmeans: {}".format(str(e)))

    logger.info('Display')
    plt.figure()

    plt.plot(np.arange(len(objective_function_with_hier_palm)), objective_function_with_hier_palm, marker="x", label="hierarchical")
    plt.plot(np.arange(len(objective_function_with_palm)), objective_function_with_palm, marker="x", label="palm")
    plt.plot(np.arange(len(objective_values_k)), objective_values_k, marker="x", label="kmeans")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
