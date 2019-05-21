import unittest
import numpy as np
from scipy.linalg import hadamard
from pyqalm.qalm import PALM4MSA as palm4msa_fast0
from pyqalm.qalm import palm4msa_fast1, palm4msa_fast2, palm4msa_fast3
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

    def test_palm4msa_compare01(self):
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
            out0 = palm4msa_fast0(
                X,
                lst_S_init=lst_S_init,
                nb_factors=nb_factors,
                lst_projection_functions=lst_projection_functions,
                f_lambda_init=f_lambda_init,
                nb_iter=nb_iter,
                update_right_to_left=update_right_to_left,
                graphical_display=graphical_display)

            out1 = palm4msa_fast1(
                X,
                lst_S_init=lst_S_init,
                nb_factors=nb_factors,
                lst_projection_functions=lst_projection_functions,
                f_lambda_init=f_lambda_init,
                nb_iter=nb_iter,
                update_right_to_left=update_right_to_left,
                graphical_display=graphical_display)

            np.testing.assert_almost_equal(out1[0], out0[0],
                                           err_msg='f_lambda')
            self.assertEqual(len(out0[1]), nb_factors, msg='nb factors')
            self.assertEqual(out1[1].n_factors, nb_factors, msg='nb factors')
            for j in range(nb_factors):
                np.testing.assert_array_almost_equal(
                    out1[1].get_factor(j).toarray(),
                    out0[1][j],
                    err_msg='Factor {}'.format(j))
            np.testing.assert_array_almost_equal(out1[2], out0[2],
                                                 err_msg='arr_X_curr')
            np.testing.assert_equal(out1[4], out0[4], err_msg='i_iter')
            # FIXME the test on the objective function values fails
            np.testing.assert_array_almost_equal(out1[3], out0[3],
                                                 err_msg='objective_function')

    def test_palm4msa_compare12(self):
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

            out1 = palm4msa_fast1(
                X,
                lst_S_init=lst_S_init,
                nb_factors=nb_factors,
                lst_projection_functions=lst_projection_functions,
                f_lambda_init=f_lambda_init,
                nb_iter=nb_iter,
                update_right_to_left=update_right_to_left,
                graphical_display=graphical_display)

            out2 = palm4msa_fast2(
                X,
                lst_S_init=lst_S_init,
                nb_factors=nb_factors,
                lst_projection_functions=lst_projection_functions,
                f_lambda_init=f_lambda_init,
                nb_iter=nb_iter,
                update_right_to_left=update_right_to_left,
                graphical_display=graphical_display)

            np.testing.assert_almost_equal(out2[0], out1[0],
                                           err_msg='f_lambda')
            np.testing.assert_array_almost_equal(out2[2], out1[2],
                                                 err_msg='arr_X_curr')
            np.testing.assert_equal(out2[4], out1[4], err_msg='i_iter')
            np.testing.assert_array_almost_equal(out2[3], out1[3],
                                                 err_msg='objective_function')

    def test_palm4msa_compare02(self):
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

            out0 = palm4msa_fast0(
                X,
                lst_S_init=lst_S_init,
                nb_factors=nb_factors,
                lst_projection_functions=lst_projection_functions,
                f_lambda_init=f_lambda_init,
                nb_iter=nb_iter,
                update_right_to_left=update_right_to_left,
                graphical_display=graphical_display)

            out2 = palm4msa_fast2(
                X,
                lst_S_init=lst_S_init,
                nb_factors=nb_factors,
                lst_projection_functions=lst_projection_functions,
                f_lambda_init=f_lambda_init,
                nb_iter=nb_iter,
                update_right_to_left=update_right_to_left,
                graphical_display=graphical_display)

            np.testing.assert_almost_equal(out2[0], out0[0],
                                           err_msg='f_lambda')
            np.testing.assert_array_almost_equal(out2[2], out0[2],
                                                 err_msg='arr_X_curr')
            np.testing.assert_equal(out2[4], out0[4], err_msg='i_iter')
            np.testing.assert_array_almost_equal(out2[3], out0[3],
                                                 err_msg='objective_function')

    def test_palm4msa_compare23(self):
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

            out2 = palm4msa_fast2(
                X,
                lst_S_init=lst_S_init,
                nb_factors=nb_factors,
                lst_projection_functions=lst_projection_functions,
                f_lambda_init=f_lambda_init,
                nb_iter=nb_iter,
                update_right_to_left=update_right_to_left,
                graphical_display=graphical_display)

            out3 = palm4msa_fast3(
                X,
                lst_S_init=lst_S_init,
                nb_factors=nb_factors,
                lst_projection_functions=lst_projection_functions,
                f_lambda_init=f_lambda_init,
                nb_iter=nb_iter,
                update_right_to_left=update_right_to_left,
                graphical_display=graphical_display)

            np.testing.assert_almost_equal(out3[0], out2[0],
                                           err_msg='f_lambda')
            np.testing.assert_array_almost_equal(out3[2], out2[2],
                                                 err_msg='arr_X_curr')
            np.testing.assert_equal(out3[3], out2[4], err_msg='i_iter')


if __name__ == '__main__':
    unittest.main()
