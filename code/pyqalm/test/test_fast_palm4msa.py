import unittest
import numpy as np
from scipy.linalg import hadamard
from pyqalm.palm.qalm import palm4msa as palm4msa_fast0
from pyqalm.palm.qalm import hierarchical_palm4msa as hierarchical_palm4msa_slow
from pyqalm.palm.qalm_fast import palm4msa_fast1, palm4msa_fast2, palm4msa_fast3
from pyqalm.palm.qalm_fast import hierarchical_palm4msa as \
    hierarchical_palm4msa_fast
from pyqalm.utils import get_lambda_proxsplincol


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
            self.assertEqual(out2[1].n_factors, nb_factors, msg='nb factors')
            for j in range(nb_factors):
                np.testing.assert_array_almost_equal(
                    out2[1].get_factor(j).toarray(),
                    out1[1].get_factor(j).toarray(),
                    err_msg='Factor {}'.format(j))
            np.testing.assert_array_almost_equal(out2[2], out1[2],
                                                 err_msg='arr_X_curr')
            np.testing.assert_equal(out2[4], out1[4], err_msg='i_iter')
            np.testing.assert_array_almost_equal(out2[3], out1[3],
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
                graphical_display=graphical_display,
                track_objective=False)

            np.testing.assert_almost_equal(out3[0], out2[0],
                                           err_msg='f_lambda')
            self.assertEqual(out3[1].n_factors, nb_factors, msg='nb factors')
            for j in range(nb_factors):
                np.testing.assert_array_almost_equal(
                    out3[1].get_factor(j).toarray(),
                    out2[1].get_factor(j).toarray(),
                    err_msg='Factor {}'.format(j))
            np.testing.assert_array_almost_equal(out3[2], out2[2],
                                                 err_msg='arr_X_curr')
            np.testing.assert_equal(out3[4], out2[4], err_msg='i_iter')

            out3 = palm4msa_fast3(
                X,
                lst_S_init=lst_S_init,
                nb_factors=nb_factors,
                lst_projection_functions=lst_projection_functions,
                f_lambda_init=f_lambda_init,
                nb_iter=nb_iter,
                update_right_to_left=update_right_to_left,
                graphical_display=graphical_display,
                track_objective=True)

            np.testing.assert_almost_equal(out3[0], out2[0],
                                           err_msg='f_lambda')
            self.assertEqual(out3[1].n_factors, nb_factors, msg='nb factors')
            for j in range(nb_factors):
                np.testing.assert_array_almost_equal(
                    out3[1].get_factor(j).toarray(),
                    out2[1].get_factor(j).toarray(),
                    err_msg='Factor {}'.format(j))
            np.testing.assert_array_almost_equal(out3[2], out2[2],
                                                 err_msg='arr_X_curr')
            np.testing.assert_equal(out3[4], out2[4], err_msg='i_iter')
            np.testing.assert_array_almost_equal(out3[3], out2[3],
                                                 err_msg='objective_function')

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
                graphical_display=False,
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

            np.testing.assert_almost_equal(out1[0], out0[0],
                                           err_msg='lambda')

            self.assertEqual(out1[1].n_factors, nb_factors, msg='nb factors')
            for j in range(nb_factors):
                np.testing.assert_array_almost_equal(
                    out1[1].get_factor(j).toarray(), out0[1][j],
                    err_msg='Factor {}'.format(j))
            np.testing.assert_array_almost_equal(out1[2], out0[2], err_msg='X')
            np.testing.assert_equal(out1[3], out0[3], err_msg='nb_iter_by_factor')
            np.testing.assert_array_almost_equal(out1[4], out0[4],
                                                 err_msg='objective_function')
            # print(out0[4])
            # np.testing.assert_array_equal(out1[3], out0[3],
            #                               err_msg='nb_iter_by_factor')


if __name__ == '__main__':
    unittest.main()
