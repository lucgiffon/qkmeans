"""
kmeans algorithm inspired from https://jonchar.net/notebooks/k-means/ .

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon

"""
import logging
import time

import daiquiri
import copy
from collections import OrderedDict
from pprint import pformat

import numpy as np
from qkmeans.qk_means.utils import compute_objective, assign_points_to_clusters, build_constraint_set_smart, get_squared_froebenius_norm_line_wise, update_clusters_with_integrity_check, \
    get_squared_froebenius_norm_line_wise_batch_by_batch, update_clusters, check_cluster_integrity
from qkmeans.qk_means.kmeans import kmeans
from scipy.sparse import csr_matrix
from sklearn import datasets
import matplotlib.pyplot as plt

from qkmeans.palm.qalm_fast import hierarchical_palm4msa, \
    palm4msa
from qkmeans.test.test_qalm import visual_evaluation_palm4msa
from qkmeans.data_structures import SparseFactors
from qkmeans.utils import logger, DataGenerator


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

def qkmeans_minibatch(X_data: np.ndarray,
           K_nb_cluster: int,
           nb_iter: int,
           nb_factors: int,
           params_palm4msa: dict,
           initialization: np.ndarray,
           batch_size:int,
           hierarchical_inside=False,
           delta_objective_error_threshold=1e-6,
           hierarchical_init=False):

    assert K_nb_cluster == initialization.shape[0]

    logger.debug("Compute squared froebenius norm of data")
    X_data_norms = get_squared_froebenius_norm_line_wise_batch_by_batch(X_data, batch_size)

    nb_examples = X_data.shape[0]
    total_nb_of_minibatch = X_data.shape[0] // batch_size

    X_centroids_hat = copy.deepcopy(initialization)

    # ################################ INIT PALM4MSA ###############################
    logger.info("Initializing QKmeans with PALM algorithm")

    lst_factors = init_lst_factors(K_nb_cluster, X_centroids_hat.shape[1], nb_factors)
    eye_norm = np.sqrt(K_nb_cluster)


    ##########################
    # GET PARAMS OF PALM4MSA #
    ##########################
    init_lambda = params_palm4msa["init_lambda"]
    nb_iter_palm = params_palm4msa["nb_iter"]
    lst_proj_op_by_fac_step = params_palm4msa["lst_constraint_sets"]
    residual_on_right = params_palm4msa["residual_on_right"]
    delta_objective_error_threshold_inner_palm = params_palm4msa["delta_objective_error_threshold"]
    track_objective_palm = params_palm4msa["track_objective"]

    ####################
    # INIT RUN OF PALM #
    ####################

    if hierarchical_inside or hierarchical_init:
        _lambda_tmp, op_factors, _, objective_palm, array_objective_hierarchical= \
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
        _lambda_tmp, op_factors, _, objective_palm, nb_iter_palm = \
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

    # ################################################################

    lst_factors = None  # safe assignment for debug

    _lambda = _lambda_tmp / eye_norm

    objective_function = np.ones(nb_iter) * -1
    lst_all_objective_functions_palm = []
    lst_all_objective_functions_palm.append(objective_palm)

    i_iter = 0
    delta_objective_error = np.inf
    while ((i_iter < nb_iter) and (delta_objective_error > delta_objective_error_threshold)):
        logger.info("Iteration number {}/{}".format(i_iter, nb_iter))

        # Re-init palm factors for iteration
        lst_factors_ = op_factors.get_list_of_factors()
        op_centroids = SparseFactors([lst_factors_[1] * _lambda] + lst_factors_[2:])

        # Prepare next epoch
        full_count_vector = np.zeros(K_nb_cluster, dtype=int)
        full_indicator_vector = np.zeros(X_data.shape[0], dtype=int)
        objective_value_so_far = 0
        X_centroids_hat = np.zeros_like(X_centroids_hat)

        for i_minibatch, example_batch_indexes in enumerate(DataGenerator(X_data, batch_size=batch_size, return_indexes=True)):
            logger.info("Minibatch number {}/{}; Iteration number {}/{}".format(i_minibatch, total_nb_of_minibatch, i_iter, nb_iter))
            example_batch = X_data[example_batch_indexes]
            example_batch_norms = X_data_norms[example_batch_indexes]

            ##########################
            # Update centroid oracle #
            ##########################

            indicator_vector, distances = assign_points_to_clusters(example_batch, op_centroids, X_norms=example_batch_norms)
            full_indicator_vector[example_batch_indexes] = indicator_vector

            cluster_names, counts = np.unique(indicator_vector, return_counts=True)
            count_vector = np.zeros(K_nb_cluster)
            count_vector[cluster_names] = counts

            full_count_vector = update_clusters(example_batch,
                                                X_centroids_hat,
                                                K_nb_cluster,
                                                full_count_vector,
                                                count_vector,
                                                indicator_vector)

            objective_value_so_far += np.sqrt(compute_objective(example_batch, op_centroids, indicator_vector))

        objective_function[i_iter] = objective_value_so_far ** 2

        # inplace modification of X_centrois_hat and full_count_vector and full_indicator_vector
        check_cluster_integrity(X_data, X_centroids_hat, K_nb_cluster, full_count_vector, full_indicator_vector)

        #########################
        # Do palm for iteration #
        #########################

        # create the diagonal of the sqrt of those counts
        diag_counts_sqrt_normalized = csr_matrix(
            (np.sqrt(full_count_vector / nb_examples),
             (np.arange(K_nb_cluster), np.arange(K_nb_cluster))))
        diag_counts_sqrt = np.sqrt(full_count_vector)

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

        _lambda = _lambda_tmp / np.sqrt(nb_examples)

        ############################

        lst_all_objective_functions_palm.append(objective_palm)

        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1]

        # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilee

        i_iter += 1

    op_centroids = SparseFactors([lst_factors_[1] * _lambda] + lst_factors_[2:])

    return objective_function[:i_iter], op_centroids, full_indicator_vector, lst_all_objective_functions_palm


if __name__ == "__main__":
    batch_size = 10000
    nb_clust = 500
    nb_iter = 2
    nb_iter_palm = 300
    residual_on_right = True
    delta_objective_error_threshold_in_palm = 1e-4
    track_objective_in_palm = False
    sparsity_factor = 2

    X = np.memmap("/home/luc/PycharmProjects/qalm_qmeans/data/external/blobs_1_billion.dat", mode="r", dtype="float32", shape=(int(1e6), 2000))

    logger.debug("Initializing clusters")
    centroids_init = X[np.random.permutation(X.shape[0])[:nb_clust]]

    nb_fac = int(np.log2(min((X.shape[1], nb_clust))))
    lst_constraints, lst_constraints_vals = build_constraint_set_smart(
        centroids_init.shape[0], centroids_init.shape[1], nb_fac,
        sparsity_factor=sparsity_factor, residual_on_right=residual_on_right)

    palm_init = {
        "init_lambda": 1.,
        "nb_iter": nb_iter_palm,
        "lst_constraint_sets": lst_constraints,
        "residual_on_right": residual_on_right,
        "delta_objective_error_threshold": delta_objective_error_threshold_in_palm,
        "track_objective": track_objective_in_palm
    }
    start = time.time()
    logger.debug("Nb iteration: {}".format(nb_iter))
    obj, _, _, _ = qkmeans_minibatch(X_data=X,
                                  K_nb_cluster=nb_clust,
                                  nb_iter=nb_iter,
                                  nb_factors=nb_fac,
                                  params_palm4msa=palm_init,
                                  initialization=centroids_init,
                                  batch_size=batch_size)
    stop = time.time()
    plt.plot(obj)
    plt.show()
    print("It took {} s".format(stop - start))
