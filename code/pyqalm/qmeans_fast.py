"""
kmeans algorithm inspired from https://jonchar.net/notebooks/k-means/ .

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon

"""
import logging
import daiquiri
import copy
from collections import OrderedDict
from pprint import pformat

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import datasets
import matplotlib.pyplot as plt

from pyqalm.qalm_fast import hierarchical_palm4msa, \
    compute_objective_function, palm4msa
from pyqalm.test.test_qalm import visual_evaluation_palm4msa
from pyqalm.data_structures import SparseFactors
from pyqalm.utils import get_side_prod, get_lambda_proxsplincol, \
    constant_proj, logger


def get_distances(X_data, centroids):
    """
    Return the matrix of distance between each data point and each centroid.

    Parameters
    ----------
    X_data : np.ndarray [n, d]
    centroids : np.ndarray or SparseFactors [k, d]

    Returns
    -------
    np.ndarray [k, n]
    """
    if isinstance(centroids, SparseFactors):
        mat_centroids = centroids.compute_product(return_array=False)
        centroid_norms = np.linalg.norm(mat_centroids.toarray(), axis=1) ** 2
        # centroid_norms = np.sqrt(centroids.power(2).sum(axis=1))
    else:
        centroid_norms = np.linalg.norm(centroids, axis=1) ** 2

    centroid_distances = centroid_norms[:, None] - 2 * centroids @ X_data.T

    return centroid_distances.T


def compute_objective(X_data, centroids, indicator_vector):
    """
    Compute K-means objective function

    Parameters
    ----------
    X_data : np.ndarray [n, d]
    centroids : np.ndarray or SparseFactors [k, d]
    indicator_vector : np.ndarray [n]

    Returns
    -------
    float
    """
    if isinstance(centroids, SparseFactors):
        centroids = centroids.compute_product()
    return np.linalg.norm(X_data - centroids[indicator_vector, :]) ** 2


def assign_points_to_clusters(X, centroids):
    """

    Parameters
    ----------
    X : np.ndarray [n, d]
    centroids : np.ndarray or SparseFactors [k, d]

    Returns
    -------
    np.ndarray [n]
        indicator_vector
    """
    distances = get_distances(X, centroids)
    # then, Determine class membership of each point
    # by picking the closest centroid
    indicator_vector = np.argmin(distances, axis=1)
    return indicator_vector


def qmeans(X_data: np.ndarray,
           K_nb_cluster: int,
           nb_iter: int,
           nb_factors: int,
           params_palm4msa: dict,
           initialization: np.ndarray,
           hierarchical_inside=False,
           graphical_display=False,
           return_objective_function=False):
    assert K_nb_cluster == initialization.shape[0]

    nb_examples = X_data.shape[0]

    logger.info("Initializing Qmeans")

    init_lambda = params_palm4msa["init_lambda"]
    nb_iter_palm = params_palm4msa["nb_iter"]
    lst_proj_op_by_fac_step = params_palm4msa["lst_constraint_sets"]

    X_centroids_hat = copy.deepcopy(initialization)
    min_K_d = min(X_centroids_hat.shape)

    lst_factors = [np.eye(min_K_d) for _ in range(nb_factors)]

    eye_norm = np.sqrt(K_nb_cluster)
    lst_factors[0] = np.eye(K_nb_cluster) / eye_norm
    lst_factors[1] = np.eye(K_nb_cluster, min_K_d)
    lst_factors[-1] = np.zeros((min_K_d, X_centroids_hat.shape[1]))

    if graphical_display:
        lst_factors_init = copy.deepcopy(lst_factors)

    _lambda_tmp, op_factors, U_centroids, nb_iter_by_factor, objective_palm = \
        hierarchical_palm4msa(
            arr_X_target=np.eye(K_nb_cluster) @ X_centroids_hat,
            lst_S_init=lst_factors,
            lst_dct_projection_function=lst_proj_op_by_fac_step,
            f_lambda_init=init_lambda * eye_norm,
            nb_iter=nb_iter_palm,
            update_right_to_left=True,
            residual_on_right=True,
            graphical_display=False,
            return_objective_function=return_objective_function)
    lst_factors = None  # safe assignment for debug

    _lambda = _lambda_tmp / eye_norm

    if graphical_display:
        if hierarchical_inside:
            plt.figure()
            plt.yscale("log")
            plt.scatter(np.arange(len(objective_palm) * 3, step=3),
                        objective_palm[:, 0], marker="x", label="before split")
            plt.scatter(np.arange(len(objective_palm) * 3, step=3) + 1,
                        objective_palm[:, 1], marker="x", label="between")
            plt.scatter(np.arange(len(objective_palm) * 3, step=3) + 2,
                        objective_palm[:, 2], marker="x",
                        label="after finetune")
            plt.plot(np.arange(len(objective_palm) * 3),
                     objective_palm.flatten(), color="k")
            plt.legend()
            plt.show()

        # FIXME
        visual_evaluation_palm4msa(
            np.eye(K_nb_cluster) @ X_centroids_hat,
            lst_factors_init,
            [x.toarray() for x in op_factors.get_list_of_factors()],
            _lambda * op_factors.compute_product())

    if return_objective_function:
        objective_function = np.empty((nb_iter, 2))
        objective_function[0, 0] = np.inf

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while (i_iter == 0) or \
            ((i_iter < nb_iter)
             and (delta_objective_error > delta_objective_error_threshold)):

        logger.info("Iteration Qmeans {}".format(i_iter))

        # U_centroids = _lambda * multi_dot(lst_factors[1:])
        lst_factors_ = op_factors.get_list_of_factors()
        op_centroids = SparseFactors([lst_factors_[1] * _lambda]
                                     + lst_factors_[2:])

        if i_iter > 0 and return_objective_function:
            objective_function[i_iter, 0] = \
                compute_objective(X_data, op_centroids, indicator_vector)

        # # Assign all points to the nearest centroid
        # # first get distance from all points to all centroids
        # distances = get_distances(X_data, op_centroids)
        # # then, Determine class membership of each point
        # # by picking the closest centroid
        # indicator_vector = np.argmin(distances, axis=1)
        indicator_vector = assign_points_to_clusters(X_data, op_centroids)

        if return_objective_function:
            objective_function[i_iter, 1] = compute_objective(X_data,
                                                              op_centroids,
                                                              indicator_vector)

        # Update centroid location using the newly
        # assigned data point classes
        for c in range(K_nb_cluster):
            X_centroids_hat[c] = np.mean(X_data[indicator_vector == c], 0)

        # get the number of observation in each cluster
        cluster_names, counts = np.unique(indicator_vector, return_counts=True)
        cluster_names_sorted = np.argsort(cluster_names)

        if len(counts) < K_nb_cluster:
            raise ValueError(
                "Some clusters have no point. Aborting iteration {}"
                .format(i_iter))

        # diag_counts_sqrt = np.diag(np.sqrt(counts[cluster_names_sorted]))  # todo use sparse matrix object
        # diag_counts_sqrt_norm = np.linalg.norm(diag_counts_sqrt)  # todo analytic sqrt(n) instead of cumputing it with norm
        # diag_counts_sqrt_normalized = diag_counts_sqrt / diag_counts_sqrt_norm
        # analytic sqrt(n) instead of cumputing it with norm
        diag_counts_sqrt_normalized = csr_matrix(
            (np.sqrt(counts[cluster_names_sorted] / nb_examples),
             (np.arange(K_nb_cluster), np.arange(K_nb_cluster))))
        diag_counts_sqrt = np.sqrt(counts[cluster_names_sorted])
        # set it as first factor
        # lst_factors[0] = diag_counts_sqrt_normalized
        op_factors.set_factor(0, diag_counts_sqrt_normalized)

        if graphical_display:
            # lst_factors_init = copy.deepcopy(lst_factors)
            lst_factors_init = [x.toarray()
                                for x in op_factors.get_list_of_factors()]

        if return_objective_function:
            loss_palm_before = compute_objective_function(
                diag_counts_sqrt[:, None,] * X_centroids_hat,
                _lambda * np.sqrt(nb_examples),
                op_factors)
            logger.info("Loss palm before: {}".format(loss_palm_before))

        if hierarchical_inside:
            _lambda_tmp, op_factors, _, nb_iter_by_factor, objective_palm = \
                hierarchical_palm4msa(
                    arr_X_target=diag_counts_sqrt[:, None,] *  X_centroids_hat,
                    lst_S_init=op_factors.get_list_of_factors(),
                    lst_dct_projection_function=lst_proj_op_by_fac_step,
                    # f_lambda_init=_lambda,
                    f_lambda_init=_lambda * np.sqrt(nb_examples),
                    nb_iter=nb_iter_palm,
                    update_right_to_left=True,
                    residual_on_right=True,
                    graphical_display=False,
                    return_objective_function=return_objective_function)

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
                         graphical_display=False,
                         track_objective=return_objective_function)

        if return_objective_function:
            loss_palm_after = compute_objective_function(
                diag_counts_sqrt[:, None,] * X_centroids_hat, _lambda_tmp, op_factors)
            logger.info("Loss palm after: {}".format(loss_palm_after))
            logger.info("Loss palm inside: {}".format(objective_palm[-1, 0]))

        if graphical_display:
            if hierarchical_inside:
                plt.figure()
                plt.yscale("log")
                plt.scatter(np.arange(len(objective_palm) * 3, step=3),
                            objective_palm[:, 0], marker="x",
                            label="before split")
                plt.scatter(np.arange(len(objective_palm) * 3, step=3) + 1,
                            objective_palm[:, 1], marker="x", label="between")
                plt.scatter(np.arange(len(objective_palm) * 3, step=3) + 2,
                            objective_palm[:, 2], marker="x",
                            label="after finetune")
                plt.plot(np.arange(len(objective_palm) * 3),
                         objective_palm.flatten(), color="k")
                plt.legend()
                plt.show()

            # FIXME
            visual_evaluation_palm4msa(
                diag_counts_sqrt[:, None,] * X_centroids_hat,
                lst_factors_init,
                [x.toarray() for x in op_factors.get_list_of_factors()],
                _lambda_tmp * op_factors.compute_product())
            # visual_evaluation_palm4msa(diag_counts_sqrt @ X_centroids_hat,
            #                            lst_factors_init, lst_factors,
            #                            _lambda_tmp * multi_dot(lst_factors))

        _lambda = _lambda_tmp / np.sqrt(nb_examples)
        # _lambda = _lambda_tmp

        if return_objective_function:
            logger.debug("Returned loss (with diag) palm: {}"
                         .format(objective_palm[-1, 0]))

        obj_fun = compute_objective(X_data, op_centroids, indicator_vector)
        if i_iter >= 1:
            if obj_fun_prev == 0:
                delta_objective_error = 0
            else:
                delta_objective_error = \
                    np.abs(obj_fun - obj_fun_prev) / obj_fun_prev
        #     delta_objective_error = np.abs(
        #         objective_function[i_iter, 0] - objective_function[
        #             i_iter - 1, 0]) / objective_function[
        #                                 i_iter - 1, 0]  # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée
        obj_fun_prev = obj_fun

        i_iter += 1

    if return_objective_function:
        return op_centroids, objective_function[:i_iter]
    else:
        return op_centroids, None


def kmeans(X_data, K_nb_cluster, nb_iter, initialization):
    plt.figure()
    # plt.yscale("log")

    # Initialize our centroids by picking random data points
    U_centroids_hat = copy.deepcopy(initialization)
    U_centroids = U_centroids_hat

    objective_function = np.empty((nb_iter,))

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while (i_iter == 0) or ((i_iter < nb_iter) and (
            delta_objective_error > delta_objective_error_threshold)):

        logger.info("Iteration Kmeans {}".format(i_iter))

        # Assign all points to the nearest centroid
        # first get distance from all points to all centroids
        distances = get_distances(X_data, U_centroids)
        # then, Determine class membership of each point
        # by picking the closest centroid
        indicator_vector = np.argmin(distances, axis=1)

        objective_function[i_iter,] = compute_objective(X_data, U_centroids,
                                                        indicator_vector)

        # Update centroid location using the newly
        # assigned data point classes
        for c in range(K_nb_cluster):
            U_centroids_hat[c] = np.mean(X_data[indicator_vector == c], 0)

        U_centroids = U_centroids_hat

        if np.isnan(U_centroids_hat).any():
            exit("Some clusters have no point. Aborting iteration {}".format(
                i_iter))

        if i_iter >= 1:
            delta_objective_error = np.abs(
                objective_function[i_iter] - objective_function[i_iter - 1]) / \
                                    objective_function[
                                        i_iter - 1]  # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée

        i_iter += 1

    return objective_function[:i_iter], U_centroids


def build_constraint_sets(left_dim, right_dim, nb_factors, sparsity_factor):
    """
    Build constraint set for factors with first factor constant.

    :param left_dim:
    :param right_dim:
    :param nb_factors:
    :param sparsity_factor:
    :return:
    """

    inner_factor_dim = min(left_dim, right_dim)

    lst_proj_op_by_fac_step = []
    lst_proj_op_desc_by_fac_step = []

    nb_keep_values = sparsity_factor * inner_factor_dim  # sparsity factor = 5
    for k in range(nb_factors - 1):
        nb_values_residual = max(nb_keep_values, int(inner_factor_dim / 2 ** (
            k)) * inner_factor_dim)  # k instead of (k+1) for the first, constant matrix
        if k == 0:
            dct_step_lst_proj_op = {
                "split": [constant_proj, lambda mat: mat],
                "finetune": [constant_proj, lambda mat: mat]
            }
            dct_step_lst_nb_keep_values = {
                "split": ["constant_proj", "ident"],
                "finetune": ["constant_proj", "ident"]
            }
        else:
            dct_step_lst_proj_op = {
                "split": [get_lambda_proxsplincol(nb_keep_values),
                          get_lambda_proxsplincol(nb_values_residual)],
                "finetune": [constant_proj] + [
                    get_lambda_proxsplincol(nb_keep_values)] * (k) + [
                                get_lambda_proxsplincol(nb_values_residual)]
            }

            dct_step_lst_nb_keep_values = {
                "split": [nb_keep_values, nb_values_residual],
                "finetune": ["constant_proj"] + [nb_keep_values] * (k) + [
                    nb_values_residual]
            }

        lst_proj_op_by_fac_step.append(dct_step_lst_proj_op)
        lst_proj_op_desc_by_fac_step.append(dct_step_lst_nb_keep_values)

    return lst_proj_op_by_fac_step, lst_proj_op_desc_by_fac_step


def init_factors(left_dim, right_dim, nb_factors):
    inner_factor_dim = min(right_dim, left_dim)

    lst_factors = [np.eye(inner_factor_dim) for _ in range(nb_factors)]
    lst_factors[0] = np.eye(left_dim)
    lst_factors[1] = np.eye(left_dim, inner_factor_dim)
    lst_factors[-1] = np.zeros((inner_factor_dim, right_dim))
    return lst_factors


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

    sparsity_factor = 2
    nb_iter_palm = 300

    lst_constraints, lst_constraints_vals = build_constraint_sets(
        U_centroids_hat.shape[0], U_centroids_hat.shape[1], nb_factors,
        sparsity_factor=sparsity_factor)
    logger.info("constraints: {}".format(pformat(lst_constraints_vals)))

    hierarchical_palm_init = {
        "init_lambda": 1.,
        "nb_iter": nb_iter_palm,
        "lst_constraint_sets": lst_constraints}

    # try:
    graphical_display = False
    logger.info('Running QuicK-means with H-Palm')
    op_centroids_hier, objective_function_hier = \
        qmeans(X, nb_clusters, nb_iter_kmeans,
               nb_factors, hierarchical_palm_init,
               initialization=U_centroids_hat,
               graphical_display=graphical_display,
               hierarchical_inside=True,
               return_objective_function=True)

    logger.info('Running QuicK-means with Palm')
    op_centroids_palm, objective_function_palm = \
        qmeans(X, nb_clusters, nb_iter_kmeans, nb_factors,
               hierarchical_palm_init,
               initialization=U_centroids_hat,
               graphical_display=graphical_display,
               return_objective_function=True)
    # except Exception as e:
    #     logger.info("There have been a problem in qmeans: {}".format(str(e)))
    try:
        logger.info('Running K-means')
        objective_values_k, centroids_finaux = \
            kmeans(X, nb_clusters, nb_iter_kmeans,
                   initialization=U_centroids_hat)
    except SystemExit as e:
        logger.info("There have been a problem in kmeans: {}".format(str(e)))

    logger.info('Display')
    plt.figure()
    # plt.yscale("log")

    plt.scatter(np.arange(len(objective_function_palm) - 1) + 0.5,
                objective_function_palm[1:, 0], marker="x",
                label="qmeans after palm(0)", color="b")
    plt.scatter((2 * np.arange(len(objective_function_palm)) + 1) / 2 - 0.5,
                objective_function_palm[:, 1], marker="x",
                label="qmeans after t (1)", color="r")
    plt.plot(np.arange(len(objective_function_palm[:, :2].flatten()) - 1) / 2,
             np.vstack(
                 [np.array([objective_function_palm[0, 1]])[:, np.newaxis],
                  objective_function_palm[1:, :2].flatten()[:, np.newaxis]]),
             color="k", label="qmeans")
    plt.scatter(np.arange(len(objective_function_hier) - 1) + 0.5,
                objective_function_hier[1:, 0], marker="x",
                label="qmeans after palm(0)", color="b")
    plt.scatter((2 * np.arange(len(objective_function_hier)) + 1) / 2 - 0.5,
                objective_function_hier[:, 1], marker="x",
                label="qmeans after t (1)", color="r")
    plt.plot(np.arange(len(objective_function_hier[:, :2].flatten()) - 1) / 2,
             np.vstack(
                 [np.array([objective_function_hier[0, 1]])[:, np.newaxis],
                  objective_function_hier[1:, :2].flatten()[:, np.newaxis]]),
             color="c", label="qmeans hier")

    plt.plot(np.arange(len(objective_values_k)), objective_values_k,
             label="kmeans", color="g", marker="x")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
