import unittest
import numpy as np
from scipy.linalg import hadamard
from pyqalm.qalm import PALM4MSA, HierarchicalPALM4MSA, palm4msa_fast1
from pyqalm.utils import get_lambda_proxsplincol
#  , get_lambda_proxsplincol_fast


class TestPalm4Msa(unittest.TestCase):
    def setUp(self):
        self.data = dict()
        self.data['hadamard'] = hadamard(32)

        n_rows = 64
        n_cols = 77
        X = np.random.randn(n_rows, n_cols)
        self.data['random matrix'] = X

    def test_palm4msa(self):
        for k, X in self.data.items():
            print(k)

            d = np.min(X.shape)
            if X.shape[1] == d:
                X = X.T
            nb_factors = int(np.log2(d))
            lst_S_init = []
            for _ in range(nb_factors - 1):
                lst_S_init.append(np.eye(d))
            lst_S_init.append(np.zeros(X.shape))

            nb_keep_values = 2 * d
            nb_values_residual = int(d / 2 ** nb_factors) * d
            lst_projection_functions = \
                [get_lambda_proxsplincol(nb_keep_values)] * nb_factors \
                + [get_lambda_proxsplincol(nb_values_residual)]

            f_lambda_init = 1
            nb_iter = 10
            update_right_to_left = True
            graphical_display = False
            f_lambda_ref, lst_S_ref, arr_X_curr_ref, objective_function_ref, \
            i_iter_ref = \
                PALM4MSA(X,
                         lst_S_init=lst_S_init,
                         nb_factors=nb_factors,
                         lst_projection_functions=lst_projection_functions,
                         f_lambda_init=f_lambda_init,
                         nb_iter=nb_iter,
                         update_right_to_left=update_right_to_left,
                         graphical_display=graphical_display)

            # lst_projection_functions_fast = \
            #     [get_lambda_proxsplincol_fast(nb_keep_values)] * nb_factors \
            #     + [get_lambda_proxsplincol_fast(nb_values_residual)]
            f_lambda, lst_S, arr_X_curr, objective_function, i_iter = \
                palm4msa_fast1(X,
                               lst_S_init=lst_S_init,
                               nb_factors=nb_factors,
                               lst_projection_functions=lst_projection_functions,
                               # lst_projection_functions=lst_projection_functions_fast,
                               f_lambda_init=f_lambda_init,
                               nb_iter=nb_iter,
                               update_right_to_left=update_right_to_left,
                               graphical_display=graphical_display)

            np.testing.assert_almost_equal(f_lambda, f_lambda_ref)
            np.testing.assert_array_almost_equal(arr_X_curr, arr_X_curr_ref)
            np.testing.assert_equal(i_iter, i_iter_ref)
            np.testing.assert_array_almost_equal(objective_function,
                                                 objective_function_ref)


if __name__ == '__main__':
    unittest.main()
