# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""

import numpy as np
from sklearn import datasets
from pprint import pformat

from pyqalm.qk_means.utils import build_constraint_set_smart
from pyqalm.qk_means.kmeans import kmeans
from pyqalm.qk_means.qmeans_fast import qmeans


def main():
    # Main code
    np.random.seed(0)
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
    residual_on_right = nb_clusters < n_features

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
    # logger.info('Running QuicK-means with H-Palm')
    # objective_function_hier, op_centroids_hier, indicator_hier = \
    #     qmeans(X, nb_clusters, nb_iter_kmeans,
    #            nb_factors, hierarchical_palm_init,
    #            initialization=U_centroids_hat,
    #            graphical_display=graphical_display,
    #            hierarchical_inside=True)
    # # return_objective_function=True)

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


if __name__ == '__main__':
    # See https://stackoverflow.com/questions/23885147/how-do-i-use-line-profiler-from-robert-kern
    import logging
    import line_profiler
    from pyqalm.utils import logger
    from pyqalm.data_structures import SparseFactors
    from pyqalm.palm.qalm_fast import palm4msa_fast4, hierarchical_palm4msa, \
        compute_objective_function

    logger.setLevel(logging.ERROR)

    lp = line_profiler.LineProfiler()
    # Add functions to be profiled
    lp.add_function(kmeans)
    lp.add_function(qmeans)
    lp.add_function(compute_objective_function)
    lp.add_function(hierarchical_palm4msa)
    lp.add_function(palm4msa_fast4)
    lp.add_function(SparseFactors.compute_spectral_norm)
    lp.add_function(qmeans)
    lp.add_function(main)
    # Set function to run
    lp_wrapper = lp(main)

    # Run
    lp_wrapper()

    lp.print_stats()

    stats_file = 'profile_qmeans.lprof'
    lp.dump_stats(stats_file)
