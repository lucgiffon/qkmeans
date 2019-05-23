import unittest

import numpy as np
from sklearn import datasets

from pyqalm.qk_means.qmeans import qmeans as qmeans_slow
from pyqalm.qk_means.utils import build_constraint_sets
from pyqalm.qk_means.qmeans_fast import qmeans as qmeans_fast


class TestCompareQmeans(unittest.TestCase):
    def test_compare_qmeans(self):
        nb_clusters = 9
        nb_iter_kmeans = 10
        X, _ = datasets.make_blobs(n_samples=200, n_features=16, centers=32)
        U_centroids_hat = X[np.random.permutation(X.shape[0])[
                            :nb_clusters]]  # kmeans++ initialization is not feasible because complexity is O(ndk)...

        nb_factors = 5
        sparsity_factor = 3
        nb_iter_palm = 100

        lst_constraints, lst_constraints_vals = build_constraint_sets(
            U_centroids_hat.shape[0], U_centroids_hat.shape[1], nb_factors,
            sparsity_factor=sparsity_factor)
        # logger.info("constraints: {}".format(pformat(lst_constraints_vals)))

        hierarchical_palm_init = {
            "init_lambda": 1.,
            "nb_iter": nb_iter_palm,
            "lst_constraint_sets": lst_constraints}
        for hierarchical_inside in (True, False):
            print(hierarchical_inside)
            op_centroids, objective_values_q_fast = \
                qmeans_fast(X, nb_clusters, nb_iter_kmeans, nb_factors,
                            hierarchical_palm_init,
                            initialization=U_centroids_hat,
                            graphical_display=False,
                            hierarchical_inside=hierarchical_inside,
                            return_objective_function=True)
            objective_values_q_slow = \
                qmeans_slow(X, nb_clusters, nb_iter_kmeans, nb_factors,
                            hierarchical_palm_init,
                            initialization=U_centroids_hat,
                            graphical_display=False,
                            hierarchical_inside=hierarchical_inside)
            np.testing.assert_array_almost_equal(objective_values_q_fast,
                                                 objective_values_q_slow)


if __name__ == '__main__':
    unittest.main()
