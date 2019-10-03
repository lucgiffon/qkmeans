import unittest
import numpy as np
from scipy.linalg import hadamard
from qkmeans.palm.palm import hierarchical_palm4msa as hierarchical_palm4msa_slow
from qkmeans.palm.palm_fast import hierarchical_palm4msa as \
    hierarchical_palm4msa_fast
from qkmeans.utils import get_lambda_proxsplincol

import logging
from qkmeans.utils import logger
logger.setLevel(logging.INFO)


class TestPalm4Msa(unittest.TestCase):
    def setUp(self):
        self.data = dict()
        self.data['hadamard'] = hadamard(32)

        n_rows = 64
        n_cols = 77
        X = np.random.randn(n_rows, n_cols)
        self.data['random matrix'] = X

    def test_hierarchical_palm4msa_compare(self):
        for k, X in self.data.items():
            print(k)

            d = np.min(X.shape)
            if X.shape[1] == d:
                X = X.T
            nb_factors = int(np.log2(d))

            nb_iter = 300

            lst_factors = []
            for _ in range(nb_factors - 1):
                lst_factors.append(np.eye(d))
            lst_factors.append(np.zeros(X.shape))
            _lambda = 1.
            # had = hadamard(d)
            # H = had / np.sqrt(32)


            lst_proj_op_by_fac_step = []
            nb_keep_values = 2 * d
            for k in range(nb_factors - 1):
                nb_values_residual = int(d / 2 ** (k + 1)) * d
                dct_step_lst_nb_keep_values = {
                    "split": [get_lambda_proxsplincol(nb_keep_values),
                              get_lambda_proxsplincol(nb_values_residual)],
                    "finetune": [get_lambda_proxsplincol(nb_keep_values)] * (
                            k + 1) + [
                                    get_lambda_proxsplincol(nb_values_residual)]
                }
                lst_proj_op_by_fac_step.append(dct_step_lst_nb_keep_values)

            out1 = hierarchical_palm4msa_fast(
                arr_X_target=X,
                lst_S_init=lst_factors,
                lst_dct_projection_function=lst_proj_op_by_fac_step,
                f_lambda_init=_lambda,
                nb_iter=nb_iter,
                update_right_to_left=True,
                residual_on_right=True,
                return_objective_function=True)

            out0 = hierarchical_palm4msa_slow(
                arr_X_target=X,
                lst_S_init=lst_factors,
                lst_dct_projection_function=lst_proj_op_by_fac_step,
                f_lambda_init=_lambda,
                nb_iter=nb_iter,
                update_right_to_left=True,
                residual_on_right=True,
                graphical_display=False)

            np.testing.assert_almost_equal((out1[0] - out0[0]) / out1[0], 0, err_msg='lambda')

            self.assertEqual(out1[1].n_factors, nb_factors, msg='nb factors')
            for j in range(nb_factors):
                np.testing.assert_array_almost_equal(
                    out1[1].get_factor(j).toarray(), out0[1][j],
                    err_msg='Factor {}'.format(j))
            np.testing.assert_almost_equal(np.linalg.norm(out1[2] - out0[2]) / np.linalg.norm(out1[2]), 0, err_msg='X')


if __name__ == '__main__':
    unittest.main()
