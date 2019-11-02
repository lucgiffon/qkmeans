import unittest

import numpy as np
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.palm.projection_operators import projection_operator, inplace_hardthreshold, prox_splincol

import logging
from qkmeans.utils import logger
logger.setLevel(logging.INFO)


class TestProjectionOperators(unittest.TestCase):
    def setUp(self):
        """
        The randomly generated matrix is:

        [[-7 -2  0  1]
         [ 5 -4 -6  6]
         [ 2 -1  7  3]
         [-5 -8 -3  4]]
        """
        np.random.seed(0)
        self.matrix = np.random.permutation(16).reshape(4, 4) - 8

    def test_projection_operator(self):
        nb_keep_values = 5
        projected_matrix = projection_operator(self.matrix, nb_keep_values)
        assert projected_matrix.min() == -8
        assert projected_matrix.max() == 7
        assert len(projected_matrix[projected_matrix != 0]) == nb_keep_values

    def test_inplace_hardthreshold(self):
        nb_keep_values = 5
        inplace_hardthreshold(self.matrix, nb_keep_values)
        assert self.matrix.min() == -8
        assert self.matrix.max() == 7
        assert len(self.matrix[self.matrix != 0]) == nb_keep_values

    def assert_keep_coeffs_are_greater_in_line(self, projected, transpose=False):
        if transpose:
            removed_coeffs_matrix = self.matrix.T - projected
        else:
            removed_coeffs_matrix = self.matrix - projected

        for lin_idx, _ in enumerate(projected):
            abs_projected_coeff_lin = np.abs(projected[lin_idx])
            abs_projected_coeff_lin_no_zero = abs_projected_coeff_lin[abs_projected_coeff_lin != 0]

            abs_removed_coeff_lin = np.abs(removed_coeffs_matrix[lin_idx])
            abs_removed_coeff_lin_no_zero = abs_removed_coeff_lin[abs_removed_coeff_lin != 0]
            assert (abs_projected_coeff_lin_no_zero >= abs_removed_coeff_lin_no_zero).all()


    def test_splincol(self):
        nb_keep_values = self.matrix.shape[0]
        projected_matrix = prox_splincol(self.matrix, nb_keep_values)
        assert all(len(lin[lin != 0]) >= 1 for lin in projected_matrix)
        assert all(len(col[col != 0]) >= 1 for col in projected_matrix.T)
        assert (self.matrix[projected_matrix != 0] == projected_matrix[projected_matrix != 0]).all()

        self.assert_keep_coeffs_are_greater_in_line(projected_matrix)
        self.assert_keep_coeffs_are_greater_in_line(projected_matrix.T, transpose=True)

    def test_build_constraint_set(self):
        left_dim = 10
        right_dim = 32
        nb_fac= 4
        sparsity_factor = 2
        residual_on_right = False

        lst_constraints, lst_constraints_vals = build_constraint_set_smart(
            left_dim, right_dim, nb_fac,
            sparsity_factor=sparsity_factor,
            residual_on_right=residual_on_right,
            constant_first=True,
            hierarchical=False)

        print(lst_constraints_vals)
if __name__ == "__main__":
    unittest.main()