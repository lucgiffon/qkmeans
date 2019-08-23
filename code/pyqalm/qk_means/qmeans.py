"""
kmeans algorithm inspired from https://jonchar.net/notebooks/k-means/ .
"""
import logging
import daiquiri
import copy
from collections import OrderedDict
from pprint import pformat

import numpy as np
from numpy.linalg import multi_dot
from pyqalm.palm.qalm import hierarchical_palm4msa, palm4msa
from pyqalm.qk_means.kmeans import kmeans
from pyqalm.qk_means.utils import build_constraint_set_smart, compute_objective, get_distances, get_squared_froebenius_norm_line_wise
from pyqalm.test.test_qalm import visual_evaluation_palm4msa
from sklearn import datasets
import matplotlib.pyplot as plt

from pyqalm.utils import logger

daiquiri.setup(level=logging.INFO)

def qmeans(X_data:np.ndarray,
           K_nb_cluster:int,
           nb_iter:int,
           nb_factors:int,
           params_palm4msa:dict,
           initialization:np.ndarray,
           hierarchical_inside=False,
           graphical_display=False):

    assert K_nb_cluster == initialization.shape[0]

    X_data_norms = get_squared_froebenius_norm_line_wise(X_data)

    init_lambda = params_palm4msa["init_lambda"]
    nb_iter_palm = params_palm4msa["nb_iter"]
    lst_proj_op_by_fac_step = params_palm4msa["lst_constraint_sets"]
    residual_on_right = params_palm4msa["residual_on_right"]

    X_centroids_hat = copy.deepcopy(initialization)
    min_K_d = min(X_centroids_hat.shape)

    lst_factors = [np.eye(min_K_d) for _ in range(nb_factors)]

    eye_norm = np.sqrt(K_nb_cluster)
    lst_factors[0] = np.eye(K_nb_cluster) / eye_norm
    lst_factors[1] = np.eye(K_nb_cluster, min_K_d)
    lst_factors[-1] = np.zeros((min_K_d, X_centroids_hat.shape[1]))

    if graphical_display:
        lst_factors_init = copy.deepcopy(lst_factors)

    _lambda_tmp, lst_factors, U_centroids, nb_iter_by_factor, objective_palm = hierarchical_palm4msa(
        arr_X_target=np.eye(K_nb_cluster) @ X_centroids_hat,
        lst_S_init=lst_factors,
        lst_dct_projection_function=lst_proj_op_by_fac_step,
        f_lambda_init=init_lambda * eye_norm,
        nb_iter=nb_iter_palm,
        update_right_to_left=True,
        residual_on_right=residual_on_right,
        graphical_display=False)

    _lambda = _lambda_tmp / eye_norm

    if graphical_display:
        if hierarchical_inside:
            plt.figure()
            plt.yscale("log")
            plt.scatter(np.arange(len(objective_palm) * 3, step=3), objective_palm[:, 0], marker="x", label="before split")
            plt.scatter(np.arange(len(objective_palm) * 3, step=3) + 1, objective_palm[:, 1], marker="x", label="between")
            plt.scatter(np.arange(len(objective_palm) * 3, step=3) + 2, objective_palm[:, 2], marker="x", label="after finetune")
            plt.plot(np.arange(len(objective_palm) * 3), objective_palm.flatten(), color="k")
            plt.legend()
            plt.show()

        visual_evaluation_palm4msa(np.eye(K_nb_cluster) @ X_centroids_hat, lst_factors_init, lst_factors, _lambda * multi_dot(lst_factors))


    objective_function = np.empty((nb_iter,2))

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while (i_iter <= 1) or ((i_iter < nb_iter) and (delta_objective_error > delta_objective_error_threshold)):

        logger.info("Iteration Qmeans {}".format(i_iter))

        U_centroids = _lambda * multi_dot(lst_factors[1:])

        if i_iter > 0:
            objective_function[i_iter, 0] = compute_objective(X_data, U_centroids, indicator_vector)

        # Assign all points to the nearest centroid
        # first get distance from all points to all centroids
        distances = get_distances(X_data, U_centroids, precomputed_data_points_norm=X_data_norms)
        # then, Determine class membership of each point
        # by picking the closest centroid
        indicator_vector = np.argmin(distances, axis=1)

        objective_function[i_iter, 1] = compute_objective(X_data, U_centroids, indicator_vector)

        # Update centroid location using the newly
        # assigned data point classes
        for c in range(K_nb_cluster):
            X_centroids_hat[c] = np.mean(X_data[indicator_vector == c], 0)

        # get the number of observation in each cluster
        cluster_names, counts = np.unique(indicator_vector, return_counts=True)
        cluster_names_sorted = np.argsort(cluster_names)

        if len(counts) < K_nb_cluster:
            raise ValueError("Some clusters have no point. Aborting iteration {}".format(i_iter))

        diag_counts_sqrt = np.diag(np.sqrt(counts[cluster_names_sorted])) # todo use sparse matrix object
        diag_counts_sqrt_norm = np.linalg.norm(diag_counts_sqrt) # todo analytic sqrt(n) instead of cumputing it with norm
        diag_counts_sqrt_normalized = diag_counts_sqrt / diag_counts_sqrt_norm
        # set it as first factor
        lst_factors[0] = diag_counts_sqrt_normalized

        if graphical_display:
            lst_factors_init = copy.deepcopy(lst_factors)

        if hierarchical_inside:
            _lambda_tmp, lst_factors, _, nb_iter_by_factor, objective_palm = hierarchical_palm4msa(
                arr_X_target=diag_counts_sqrt @ X_centroids_hat,
                lst_S_init=lst_factors,
                lst_dct_projection_function=lst_proj_op_by_fac_step,
                # f_lambda_init=_lambda,
                f_lambda_init=_lambda*diag_counts_sqrt_norm,
                nb_iter=nb_iter_palm,
                update_right_to_left=True,
                residual_on_right=residual_on_right,
                graphical_display=False)

            loss_palm_before = objective_palm[0, 0]
            loss_palm_after = objective_palm[-1, -1]

        else:
            _lambda_tmp, lst_factors, _, objective_palm, nb_iter_palm = palm4msa(
                arr_X_target=diag_counts_sqrt @ X_centroids_hat,
                lst_S_init=lst_factors,
                nb_factors=len(lst_factors),
                lst_projection_functions=lst_proj_op_by_fac_step[-1]["finetune"],
                f_lambda_init=_lambda * diag_counts_sqrt_norm,
                nb_iter=nb_iter_palm,
                update_right_to_left=True,
                graphical_display=False)

            loss_palm_before = objective_palm[0, -1]
            loss_palm_after = objective_palm[-1, -1]

        logger.debug("Loss palm before: {}".format(loss_palm_before))
        logger.debug("Loss palm after: {}".format(loss_palm_after))

        if graphical_display:
            if hierarchical_inside:
                plt.figure()
                plt.yscale("log")
                plt.scatter(np.arange(len(objective_palm) * 3, step=3), objective_palm[:, 0], marker="x", label="before split")
                plt.scatter(np.arange(len(objective_palm) * 3, step=3) + 1, objective_palm[:, 1], marker="x", label="between")
                plt.scatter(np.arange(len(objective_palm) * 3, step=3) + 2, objective_palm[:, 2], marker="x", label="after finetune")
                plt.plot(np.arange(len(objective_palm) * 3), objective_palm.flatten(), color="k")
                plt.legend()
                plt.show()

            visual_evaluation_palm4msa(diag_counts_sqrt @ X_centroids_hat, lst_factors_init, lst_factors, _lambda_tmp * multi_dot(lst_factors))

        _lambda = _lambda_tmp / diag_counts_sqrt_norm
        # _lambda = _lambda_tmp

        logger.debug("Returned loss (with diag) palm: {}".format(objective_palm[-1, 0]))

        if i_iter >= 2:
            delta_objective_error = np.abs(objective_function[i_iter, 0] - objective_function[i_iter-1, 0]) / objective_function[i_iter-1, 0] # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée

        i_iter += 1

    U_centroids = _lambda * multi_dot(lst_factors[1:])
    distances = get_distances(X_data, U_centroids, precomputed_data_points_norm=X_data_norms)
    indicator_vector = np.argmin(distances, axis=1)

    return objective_function[:i_iter], U_centroids, indicator_vector


if __name__ == '__main__':

    np.random.seed(0)
    daiquiri.setup(level=logging.INFO)
    nb_clusters = 10
    nb_iter_kmeans = 10
    X, _ = datasets.make_blobs(n_samples=100000, n_features=20, centers=5000)
    U_centroids_hat = X[np.random.permutation(X.shape[0])[:nb_clusters]] # kmeans++ initialization is not feasible because complexity is O(ndk)...

    nb_factors = 5
    sparsity_factor = 2
    nb_iter_palm = 300

    residual_on_right = False

    # lst_constraints, lst_constraints_vals = build_constraint_sets(U_centroids_hat.shape[0], U_centroids_hat.shape[1], nb_factors, sparsity_factor=sparsity_factor)
    K = U_centroids_hat.shape[0]
    d = U_centroids_hat.shape[1]
    lst_constraints, lst_constraints_vals = build_constraint_set_smart(K, d, nb_factors, sparsity_factor=sparsity_factor, residual_on_right=residual_on_right)
    logger.info("constraints: {}".format(pformat(lst_constraints_vals)))

    hierarchical_palm_init = {
        "init_lambda": 1.,
        "nb_iter": nb_iter_palm,
        "lst_constraint_sets": lst_constraints,
        "residual_on_right": residual_on_right}

    # try:
    objective_values_q_hier, centroids_finaux_q_hier, indicator_hier = qmeans(X, nb_clusters, nb_iter_kmeans, nb_factors, hierarchical_palm_init, initialization=U_centroids_hat, graphical_display=True, hierarchical_inside=True)
    objective_values_q_hier, centroids_finaux_q_hier, indicator = qmeans(X, nb_clusters, nb_iter_kmeans, nb_factors, hierarchical_palm_init, initialization=U_centroids_hat, graphical_display=True, hierarchical_inside=True)
    objective_values_q, centroids_finaux_q, indicator = qmeans(X, nb_clusters, nb_iter_kmeans, nb_factors, hierarchical_palm_init, initialization=U_centroids_hat, graphical_display=False)
    # except Exception as e:
    #     logger.info("There have been a problem in qmeans: {}".format(str(e)))
    try:

        objective_values_k, centroids_finaux, indicator = kmeans(X, nb_clusters, nb_iter_kmeans, initialization=U_centroids_hat)
    except SystemExit as e:
        logger.info("There have been a problem in kmeans: {}".format(str(e)))

    plt.figure()
    # plt.yscale("log")

    plt.scatter(np.arange(len(objective_values_q)-1)+0.5, objective_values_q[1:, 0], marker="x", label="qmeans after palm(0)", color="b")
    plt.scatter((2*np.arange(len(objective_values_q))+1)/2-0.5, objective_values_q[:, 1], marker="x", label="qmeans after t (1)", color="r")
    plt.plot(np.arange(len(objective_values_q[:, :2].flatten())-1)/2, np.vstack([np.array([objective_values_q[0, 1]])[:, np.newaxis], objective_values_q[1:, :2].flatten()[:, np.newaxis]]), color="k", label="qmeans")
    plt.scatter(np.arange(len(objective_values_q_hier) - 1) + 0.5, objective_values_q_hier[1:, 0], marker="x", label="qmeans after palm(0)", color="b")
    plt.scatter((2 * np.arange(len(objective_values_q_hier)) + 1) / 2 - 0.5, objective_values_q_hier[:, 1], marker="x", label="qmeans after t (1)", color="r")
    plt.plot(np.arange(len(objective_values_q_hier[:, :2].flatten())-1)/2, np.vstack([np.array([objective_values_q_hier[0, 1]])[:, np.newaxis], objective_values_q_hier[1:, :2].flatten()[:, np.newaxis]]), color="c", label="qmeans hier")

    plt.plot(np.arange(len(objective_values_k)), objective_values_k, label="kmeans", color="g", marker="x")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()