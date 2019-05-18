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
from pyqalm.qalm import HierarchicalPALM4MSA, compute_objective_function
from pyqalm.test.test_qalm import visual_evaluation_palm4msa
from sklearn import datasets
import matplotlib.pyplot as plt

from pyqalm.utils import get_side_prod, logger, get_lambda_proxsplincol, constant_proj

daiquiri.setup(level=logging.INFO)

def initialize_clusters(K_nb_clusters, d_dimension, nb_factors):
    """
    Initialize clusters as a product of sparse matrices.

    :return: The list of the sparse matrices whose product gives the initial clusters
    """
    min_K_d = min([K_nb_clusters, d_dimension])

    lst_factors = [np.eye(min_K_d) for i in range(nb_factors) ]
    lst_factors[-1] = np.random.rand(min_K_d, d_dimension)
    lst_factors[0] = np.zeros((K_nb_clusters, min_K_d))
    lst_factors[0][np.diag_indices(min_K_d)] = np.random.rand(min_K_d)
    # dansle papier gribonval le facteur de droite est initialisé "full zero"
    # mais ça marche aussi avec n'importe quelle valeur...ce qui serait nécessaire dans notre cas
    # todo pourquoi voulaient-ils mettre des zéros?

    U_centroids = get_side_prod(lst_factors)
    return lst_factors, U_centroids

def get_distances(X_data, U_centroids):
    """
    Return the matrice of distance between each data point and each centroid.
    :param X_data:
    :param U_centroids:
    :return:
    """
    centroid_norms = np.linalg.norm(U_centroids, axis=1) ** 2
    # todo tirer parti de la sparsité des matrices.
    centroid_distances = -2*(U_centroids @ X_data.T) + centroid_norms[:, np.newaxis]

    return centroid_distances.T


def compute_objective(X_data, U_centroids, indicator_vector):
    return np.linalg.norm(X_data - U_centroids[indicator_vector]) ** 2

def qmeans(X_data, K_nb_cluster, nb_iter, nb_factors, params_palm4msa, initialization):

    plt.figure()
    # plt.yscale("log")

    init_lambda = params_palm4msa["init_lambda"]
    nb_iter_palm = params_palm4msa["nb_iter"]
    lst_proj_op_by_fac_step = params_palm4msa["lst_constraint_sets"]

    # Initialize our centroids by picking random data points
    X_centroids_hat = copy.deepcopy(initialization)
    # lst_factors, U_centroids = initialize_clusters(K_nb_cluster, X_data.shape[1], nb_factors)
    min_K_d = min(X_centroids_hat.shape)

    lst_factors = [np.eye(min_K_d) for _ in range(nb_factors)]
    lst_factors[0] = np.eye(K_nb_cluster)
    lst_factors[1] = np.eye(K_nb_cluster, min_K_d)
    lst_factors[-1] = np.zeros((min_K_d, X_centroids_hat.shape[1]))

    _lambda, lst_factors, U_centroids, nb_iter_by_factor, _ = HierarchicalPALM4MSA(
        arr_X_target=np.eye(K_nb_cluster) @ X_centroids_hat,
        lst_S_init=lst_factors,
        lst_dct_projection_function=lst_proj_op_by_fac_step,
        f_lambda_init=init_lambda,
        nb_iter=nb_iter_palm,
        update_right_to_left=True,
        residual_on_right=True,
        graphical_display=False)

    lst_factors_bis = copy.deepcopy(lst_factors)
    _lambda_tmp = _lambda
    _lambda_bis = _lambda

    objective_function = np.empty((nb_iter,4))

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while (i_iter == 0) or ((i_iter < nb_iter) and (delta_objective_error > delta_objective_error_threshold)):

        logger.info("Iteration Qmeans {}".format(i_iter))

        U_centroids = _lambda * multi_dot(lst_factors[1:])

        if i_iter > 0:
            objective_function[i_iter, 1] = compute_objective(X_data, U_centroids, indicator_vector)
            objective_function[i_iter, 3] = compute_objective(X_data, _lambda_bis * multi_dot(lst_factors_bis), indicator_vector)

        # Assign all points to the nearest centroid
        # first get distance from all points to all centroids
        distances = get_distances(X_data, U_centroids)
        # then, Determine class membership of each point
        # by picking the closest centroid
        indicator_vector = np.argmin(distances, axis=1)

        objective_function[i_iter, 0] = compute_objective(X_data, U_centroids, indicator_vector)

        # Update centroid location using the newly
        # assigned data point classes
        for c in range(K_nb_cluster):
            X_centroids_hat[c] = np.mean(X_data[indicator_vector == c], 0)

        # get the number of observation in each cluster
        cluster_names, counts = np.unique(indicator_vector, return_counts=True)
        cluster_names_sorted = np.argsort(cluster_names)
        diag_counts = np.diag(np.sqrt(counts[cluster_names_sorted])) # todo use sparse matrix object
        diag_counts_norm = np.linalg.norm(diag_counts)
        diag_counts_normalized = diag_counts / diag_counts_norm
        # set it as first factor
        lst_factors[0] = diag_counts_normalized

        objective_function[i_iter, 2] = compute_objective(X_data, X_centroids_hat, indicator_vector)

        computed_loss_palm_after = compute_objective_function(diag_counts @ X_centroids_hat, _lambda_tmp, lst_factors)
        logger.info("Computed loss reconstruction (with diag) before: {}".format(computed_loss_palm_after))

        loss_reconstruction_before_palm = compute_objective_function(X_centroids_hat, _lambda, lst_factors[1:])
        logger.info("Loss reconstruction (centroids_matrix) before: {}".format(loss_reconstruction_before_palm))
        # init_factor = copy.deepcopy(lst_factors) # for visual evaluation
        _lambda_tmp, lst_factors, _, nb_iter_by_factor, objective_palm = HierarchicalPALM4MSA(
            arr_X_target=diag_counts @ X_centroids_hat,
            lst_S_init=lst_factors,
            lst_dct_projection_function=lst_proj_op_by_fac_step,
            f_lambda_init=_lambda_tmp,
            nb_iter=nb_iter_palm,
            update_right_to_left=True,
            residual_on_right=True,
            graphical_display=False)

        _lambda_bis, lst_factors_bis, _, nb_iter_by_factor_bis, objective_palm_bis = HierarchicalPALM4MSA(
            arr_X_target=np.eye(K_nb_cluster) @ X_centroids_hat,
            lst_S_init=lst_factors_bis,
            lst_dct_projection_function=lst_proj_op_by_fac_step,
            f_lambda_init=_lambda_bis,
            nb_iter=nb_iter_palm,
            update_right_to_left=True,
            residual_on_right=True,
            graphical_display=False)

        _lambda = _lambda_tmp / diag_counts_norm
        loss_reconstruction_after_palm = compute_objective_function(X_centroids_hat, _lambda, lst_factors[1:])
        logger.info("Loss reconstruction (centroids_matrix) after: {}".format(loss_reconstruction_after_palm))

        computed_loss_palm_after = compute_objective_function(diag_counts @ X_centroids_hat, _lambda_tmp, lst_factors)
        logger.info("Computed loss reconstruction (with diag) after: {}".format(computed_loss_palm_after))
        logger.info("Returned loss (with diag) palm: {}".format(objective_palm[-1, 0]))

        # U_centroids = X_centroids_hat

        # visual_evaluation_palm4msa(diag_counts @ X_centroids_hat, init_factor, lst_factors, U_centroids)

        if np.isnan(X_centroids_hat).any():
            exit("Some clusters have no point. Aborting iteration {}".format(i_iter))

        # plt.scatter(i_iter, objective_function[i_iter])
        # plt.pause(1)
        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter, 0] - objective_function[i_iter-1, 0]) / objective_function[i_iter-1, 0] # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée

        i_iter += 1

    return objective_function[:i_iter]


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
    # while (i_iter == 0) or ((i_iter < nb_iter) and (delta_objective_error > delta_objective_error_threshold)):
    while (i_iter == 0) or ((i_iter < nb_iter) ):

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

        # plt.scatter(i_iter, objective_function[i_iter])
        # plt.pause(1)
        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1] # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée


        i_iter += 1

    return objective_function[:i_iter], U_centroids



def build_constraint_sets(right_dim, left_dim, nb_factors, sparsity_factor):

    inner_factor_dim = min(right_dim, left_dim)

    lst_proj_op_by_fac_step = []
    lst_proj_op_desc_by_fac_step = []

    nb_keep_values = sparsity_factor * inner_factor_dim  # sparsity factor = 5
    for k in range(nb_factors - 1):
        nb_values_residual = max(nb_keep_values, int(inner_factor_dim / 2 ** (k)) * inner_factor_dim)  # k instead of (k+1) for the first, constant matrix
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
                "split": [get_lambda_proxsplincol(nb_keep_values), get_lambda_proxsplincol(nb_values_residual)],
                "finetune": [constant_proj] + [get_lambda_proxsplincol(nb_keep_values)] * (k) + [get_lambda_proxsplincol(nb_values_residual)]
            }

            dct_step_lst_nb_keep_values = {
                "split": [nb_keep_values, nb_values_residual],
                "finetune": ["constant_proj"] + [nb_keep_values] * (k) + [nb_values_residual]
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
    # np.random.seed(3)

    nb_clusters = 10
    nb_iter_kmeans = 10
    X, _ = datasets.make_blobs(n_samples=1000, n_features=20, centers=50)
    U_centroids_hat = X[np.random.permutation(X.shape[0])[:nb_clusters]] # kmeans++ initialization is not feasible because complexity is O(ndk)...

    nb_factors = 5
    sparsity_factor = 10
    nb_iter_palm = 300

    lst_constraints, lst_constraints_vals = build_constraint_sets(U_centroids_hat.shape[0], U_centroids_hat.shape[1], nb_factors, sparsity_factor=sparsity_factor)
    logger.info("constraints: {}".format(pformat(lst_constraints_vals)))

    hierarchical_palm_init = {
        "init_lambda": 1.,
        "nb_iter": nb_iter_palm,
        "lst_constraint_sets": lst_constraints}

    try:
        objective_values_q = qmeans(X, nb_clusters, nb_iter_kmeans, nb_factors, hierarchical_palm_init, initialization=U_centroids_hat)
    except Exception as e:
        logger.info("There have been a problem in qmeans: {}".format(str(e)))
    try:
        objective_values_k, centroids_finaux = kmeans(X, nb_clusters, nb_iter_kmeans, initialization=U_centroids_hat)
    except SystemExit as e:
        logger.info("There have been a problem in kmeans: {}".format(str(e)))

    # todo tracer la valeur objectif avec la decomposition des centroides donnés par kmeans
    # lst_S_init = init_factors(U_centroids_hat.shape[0], U_centroids_hat.shape[1])
    # _lambda, lst_factors, _, nb_iter_by_factor = HierarchicalPALM4MSA(
    #     arr_X_target=centroids_finaux,
    #     lst_S_init=lst_S_init,
    #     lst_dct_projection_function=lst_constraints,
    #     f_lambda_init=1.,
    #     nb_iter=nb_iter_palm,
    #     update_right_to_left=True,
    #     residual_on_right=True,
    #     graphical_display=False)

    plt.figure()
    # plt.yscale("log")
    # plt.plot(objective_values_q, label="qmeans")

    plt.scatter(np.arange(len(objective_values_q)), objective_values_q[:, 0], marker="x", label="qmeans after t (0)", color="r")
    plt.scatter((2*np.arange(len(objective_values_q))+1)/2, objective_values_q[:, 1], marker="x", label="qmeans after palm (1)", color="b")
    plt.scatter((2*np.arange(len(objective_values_q))+1)/2, objective_values_q[:, 2], marker="x", label="qmeans x_hat (2)", color="y")
    plt.scatter((2*np.arange(len(objective_values_q))+1)/2, objective_values_q[:, 3], marker="x", label="qmeans after palm without diag (3)", color="c")
    plt.plot(np.arange(len(objective_values_k)), objective_values_k, label="kmeans", color="g", marker="x")
    plt.plot(np.arange(len(objective_values_q[:, :2].flatten()))/2, objective_values_q[:, :2].flatten(), color="k", label="qmeans")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()