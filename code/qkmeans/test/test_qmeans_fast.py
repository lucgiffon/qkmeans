import unittest

import numpy as np
from sklearn import datasets

from qkmeans.core.qkmeans_naive import qmeans as qmeans_slow
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.core.qmeans_fast import qmeans as qmeans_fast


class TestCompareQmeans(unittest.TestCase):
    def setUp(self):
        self.nb_clusters = 9
        self.nb_iter_kmeans = 10

        self.nb_factors = 5
        self.sparsity_factor = 3
        self.nb_iter_palm = 100
        self.residual_on_right = True
        self.delta_objective_error_threshold = 1e-6

        self.X, _ = datasets.make_blobs(n_samples=200, n_features=16, centers=32)
        self.U_centroids_hat = self.X[np.random.permutation(self.X.shape[0])[:self.nb_clusters]]

    @unittest.skip("qmeans slow is deprecated")
    def test_compare_qmeans(self):


        lst_constraints, lst_constraints_vals = build_constraint_set_smart(
            self.U_centroids_hat.shape[0], self.U_centroids_hat.shape[1], self.nb_factors,
            sparsity_factor=self.sparsity_factor,
            residual_on_right=self.residual_on_right)

        hierarchical_palm_init = {
            "init_lambda": 1.,
            "nb_iter": self.nb_iter_palm,
            "lst_constraint_sets": lst_constraints,
            "residual_on_right": self.residual_on_right}

        for hierarchical_inside in (True, False):
            print(hierarchical_inside)
            objective_values_q_fast, op_centroids, t_fast = \
                qmeans_fast(self.X, self.nb_clusters, self.nb_iter_kmeans, self.nb_factors,
                            hierarchical_palm_init,
                            initialization=self.U_centroids_hat,
                            hierarchical_inside=hierarchical_inside)
            objective_values_q_slow, U, t_slow = \
                qmeans_slow(self.X, self.nb_clusters, self.nb_iter_kmeans, self.nb_factors,
                            hierarchical_palm_init,
                            initialization=self.U_centroids_hat,
                            graphical_display=False,
                            hierarchical_inside=hierarchical_inside)
            np.testing.assert_array_almost_equal(objective_values_q_fast,
                                                 objective_values_q_slow[:, 1])
            np.testing.assert_array_almost_equal(t_fast, t_slow)

    def test_run_qmeans_fast(self):
        lst_constraints, lst_constraints_vals = build_constraint_set_smart(
            self.U_centroids_hat.shape[0], self.U_centroids_hat.shape[1], self.nb_factors,
            sparsity_factor=self.sparsity_factor,
            residual_on_right=self.residual_on_right)

        hierarchical_palm_init = {
            "init_lambda": 1.,
            "nb_iter": self.nb_iter_palm,
            "lst_constraint_sets": lst_constraints,
            "residual_on_right": self.residual_on_right,
            "delta_objective_error_threshold": self.delta_objective_error_threshold,
            "track_objective": True}

        for hierarchical_inside in (True, False):
            print(hierarchical_inside)
            objective_values_q_fast, op_centroids, t_fast, lst_obj_palm = \
                qmeans_fast(self.X,
                            self.nb_clusters,
                            self.nb_iter_kmeans,
                            self.nb_factors,
                            hierarchical_palm_init,
                            initialization=self.U_centroids_hat,
                            hierarchical_inside=hierarchical_inside)

            self.assertEqual(len(op_centroids), self.nb_factors - 1)
            self.assertTrue(len(objective_values_q_fast) <= self.nb_iter_kmeans)
            self.assertEqual(len(t_fast), len(self.X))
            self.assertEqual(len(lst_obj_palm), len(objective_values_q_fast)+1)
            if hierarchical_inside == True:
                self.assertTrue(all(len(obj_palm) == self.nb_factors - 1 for obj_palm in lst_obj_palm))
                self.assertTrue(all(all(len(sub_palm_split_fine) == 2 for sub_palm_split_fine in obj_palm) for obj_palm in lst_obj_palm))
            else:
                self.assertTrue(all(len(obj_palm) == self.nb_iter_palm for obj_palm in lst_obj_palm))

if __name__ == '__main__':
    unittest.main()
