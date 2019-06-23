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
from pyqalm.qk_means.utils import compute_objective, assign_points_to_clusters, build_constraint_set_smart
from pyqalm.qk_means.kmeans import kmeans
from scipy.sparse import csr_matrix
from sklearn import datasets
import matplotlib.pyplot as plt

from pyqalm.palm.qalm_fast import hierarchical_palm4msa, \
    palm4msa
from pyqalm.test.test_qalm import visual_evaluation_palm4msa
from pyqalm.data_structures import SparseFactors
from pyqalm.utils import get_lambda_proxsplincol, \
    constant_proj, logger

def init_lst_factors(d_in, d_out, p_nb_factors):
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
           graphical_display=False):
           # return_objective_function=False):

    assert K_nb_cluster == initialization.shape[0]

    if graphical_display:
        raise NotImplementedError("Not implemented graphical display")

    nb_examples = X_data.shape[0]

    logger.info("Initializing Qmeans")

    init_lambda = params_palm4msa["init_lambda"]
    nb_iter_palm = params_palm4msa["nb_iter"]
    lst_proj_op_by_fac_step = params_palm4msa["lst_constraint_sets"]
    residual_on_right = params_palm4msa["residual_on_right"]

    X_centroids_hat = copy.deepcopy(initialization)

    lst_factors = init_lst_factors(K_nb_cluster, X_centroids_hat.shape[1], nb_factors)

    if graphical_display:
        lst_factors_init = copy.deepcopy(lst_factors)

    eye_norm = np.sqrt(K_nb_cluster)

    _lambda_tmp, op_factors, U_centroids, nb_iter_by_factor, objective_palm = \
        hierarchical_palm4msa(
            arr_X_target=np.eye(K_nb_cluster) @ X_centroids_hat,
            lst_S_init=lst_factors,
            lst_dct_projection_function=lst_proj_op_by_fac_step,
            f_lambda_init=init_lambda * eye_norm,
            nb_iter=nb_iter_palm,
            update_right_to_left=True,
            residual_on_right=residual_on_right,
            graphical_display=False)
            # return_objective_function=False)
    lst_factors = None  # safe assignment for debug

    _lambda = _lambda_tmp / eye_norm

    if graphical_display:
        if hierarchical_inside:
        # if hierarchical_inside and return_objective_function:
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

    # if return_objective_function:
    objective_function = np.empty(nb_iter)
    # objective_function[0, 0] = np.nan

    # Loop for the maximum number of iterations
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while (i_iter <= 1) or ((i_iter < nb_iter) and (delta_objective_error > delta_objective_error_threshold)):

        logger.info("Iteration Qmeans {}".format(i_iter))

        # U_centroids = _lambda * multi_dot(lst_factors[1:])
        lst_factors_ = op_factors.get_list_of_factors()
        op_centroids = SparseFactors([lst_factors_[1] * _lambda] + lst_factors_[2:])

        # if i_iter > 0 and return_objective_function:
        #     objective_function[i_iter, 0] = \
        #         compute_objective(X_data, op_centroids, indicator_vector)

        # # Assign all points to the nearest centroid
        # # first get distance from all points to all centroids
        # distances = get_distances(X_data, op_centroids)
        # # then, Determine class membership of each point
        # # by picking the closest centroid
        # indicator_vector = np.argmin(distances, axis=1)

        indicator_vector = assign_points_to_clusters(X_data, op_centroids)

        # TODO return distances to assigned cluster in
        #  assign_points_to_clusters and used them in compute_objective to
        #  save computations
        objective_function[i_iter] = compute_objective(X_data, op_centroids, indicator_vector)

        # if return_objective_function:
        #     objective_function[i_iter, 1] = compute_objective(X_data,
        #                                                       op_centroids,
        #                                                       indicator_vector)

        # Update centroid location using the newly
        # assigned data point classes

        # get the number of observation in each cluster

        cluster_names, counts = np.unique(indicator_vector, return_counts=True)
        cluster_names_sorted = np.argsort(cluster_names)

        biggest_cluster_index = np.argmax(counts)  # type: int
        biggest_cluster = cluster_names[biggest_cluster_index]
        biggest_cluster_data = X_data[indicator_vector == biggest_cluster]

        # check if all clusters still have points
        for c in range(K_nb_cluster):
            cluster_data = X_data[indicator_vector == c]
            if len(cluster_data) == 0:
                logger.warning("cluster has lost data, add new cluster. cluster idx: {}".format(c))
                X_centroids_hat[c] = biggest_cluster_data[np.random.randint(len(biggest_cluster_data))].reshape(1, -1)
                counts = list(counts)
                counts[biggest_cluster_index] -= 1
                counts.append(1)
                counts = np.array(counts)
                cluster_names_sorted = list(cluster_names_sorted)
                cluster_names_sorted.append(c)
                cluster_names_sorted = np.array(cluster_names_sorted)
            else:
                X_centroids_hat[c] = np.mean(X_data[indicator_vector == c], 0)


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

        # if return_objective_function:
        #     loss_palm_before = compute_objective_function(
        #         diag_counts_sqrt[:, None,] * X_centroids_hat,
        #         _lambda * np.sqrt(nb_examples),
        #         op_factors)
        #     logger.info("Loss palm before: {}".format(loss_palm_before))

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
                    residual_on_right=residual_on_right,
                    graphical_display=False,
                    return_objective_function=False)

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
                         track_objective=False)

        # if return_objective_function:
        #     loss_palm_after = compute_objective_function(
        #         diag_counts_sqrt[:, None,] * X_centroids_hat, _lambda_tmp, op_factors)
        #     logger.info("Loss palm after: {}".format(loss_palm_after))
        #     logger.info("Loss palm inside: {}".format(objective_palm[-1, 0]))

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

        # if return_objective_function:
        #     logger.debug("Returned loss (with diag) palm: {}"
        #                  .format(objective_palm[-1, 0]))


        if i_iter >= 2:
            delta_objective_error = np.abs(objective_function[i_iter] - objective_function[i_iter-1]) / objective_function[i_iter-1] # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée

        # if i_iter >= 1:
        #     if obj_fun_prev == 0:
        #         delta_objective_error = 0
        #     else:
        #         delta_objective_error = \
        #             np.abs(obj_fun - obj_fun_prev) / obj_fun_prev
        #     delta_objective_error = np.abs(
        #         objective_function[i_iter, 0] - objective_function[
        #             i_iter - 1, 0]) / objective_function[
        #                                 i_iter - 1, 0]  # todo vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée
        # obj_fun_prev = obj_fun

        i_iter += 1

    op_centroids = SparseFactors([lst_factors_[1] * _lambda] + lst_factors_[2:])

    # if return_objective_function:
    return objective_function[:i_iter], op_centroids, indicator_vector
    # else:
    #     return op_centroids, None


if __name__ == '__main__':
    np.random.seed(0)
    daiquiri.setup(level=logging.INFO)
    small_dim = False
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
    nb_iter_palm = 300

    lst_constraints, lst_constraints_vals = build_constraint_set_smart(
        U_centroids_hat.shape[0], U_centroids_hat.shape[1], nb_factors,
        sparsity_factor=sparsity_factor, residual_on_right=residual_on_right)
    logger.info("constraints: {}".format(pformat(lst_constraints_vals)))


    hierarchical_palm_init = {
        "init_lambda": 1.,
        "nb_iter": nb_iter_palm,
        "lst_constraint_sets": lst_constraints,
        "residual_on_right": residual_on_right
    }

    # try:
    graphical_display = False
    logger.info('Running QuicK-means with H-Palm')
    objective_function_hier, op_centroids_hier, indicator_hier = \
        qmeans(X, nb_clusters, nb_iter_kmeans,
               nb_factors, hierarchical_palm_init,
               initialization=U_centroids_hat,
               graphical_display=graphical_display,
               hierarchical_inside=True)
               # return_objective_function=True)

    logger.info('Running QuicK-means with Palm')
    objective_function_palm, op_centroids_palm, indicator_palm = \
        qmeans(X, nb_clusters, nb_iter_kmeans, nb_factors,
               hierarchical_palm_init,
               initialization=U_centroids_hat,
               graphical_display=graphical_display)
               # return_objective_function=True)
    # except Exception as e:
    #     logger.info("There have been a problem in qmeans: {}".format(str(e)))
    try:
        logger.info('Running K-means')
        objective_values_k, centroids_finaux, indicator_kmean = \
            kmeans(X, nb_clusters, nb_iter_kmeans,
                   initialization=U_centroids_hat)
    except SystemExit as e:
        logger.info("There have been a problem in kmeans: {}".format(str(e)))

    logger.info('Display')
    plt.figure()
    # plt.yscale("log")

    plt.plot(np.arange(len(objective_function_hier)), objective_function_hier, marker="x", label="hierarchical")
    plt.plot(np.arange(len(objective_function_palm)), objective_function_palm, marker="x", label="palm")
    plt.plot(np.arange(len(objective_values_k)), objective_values_k, marker="x", label="kmeans")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
