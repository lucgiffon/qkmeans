import unittest
import numpy as np

from qkmeans.data_structures import SparseFactors


class TestSparseFactors(unittest.TestCase):

    def setUp(self) -> None:
        self.A = [np.random.randint(0, 3, size=(4, 4)).astype(float)
                  for _ in range(2)]
        self.A += [np.random.randint(0, 3, size=(4, 5))]
        self.P = np.linalg.multi_dot(self.A)
        self.S = SparseFactors(self.A)

        # TODO test other examples: complex dtype, int (bug in spectral norm)

    def test_empty(self):
        S = SparseFactors([])
        np.testing.assert_array_equal(S.shape, (0, 0))
        # TODO test other methods and parameters

    def test_shape(self):
        np.testing.assert_array_equal(
            self.S.shape,
            (self.A[0].shape[0], self.A[-1].shape[-1]))
        np.testing.assert_array_equal(self.S.shape, self.P.shape)

    def test_list_H(self):
        self.assertEqual(len(self.S._lst_factors), len(self.S._lst_factors_H))
        for i in range(self.S.n_factors):
            np.testing.assert_array_almost_equal(
                self.S._lst_factors[i].toarray(),
                np.conjugate(self.S._lst_factors_H[-i-1].toarray().T),
                err_msg='Factor #{}'.format(i)
            )

    def test_product(self):
        np.testing.assert_array_almost_equal(self.P, self.S.compute_product())

    def test_matrix_vector_product(self):
        x = np.random.randn(self.S.shape[1])[:, None]
        Px = self.P @ x

        Sx = self.S.dot(x)
        np.testing.assert_array_almost_equal(Px, Sx)

        Sx = self.S @ x
        np.testing.assert_array_almost_equal(Px, Sx)

    def test_adjoint(self):
        A = self.S.adjoint()
        np.testing.assert_array_almost_equal(np.conjugate(self.P.T),
                                             A.compute_product())

    def test_transpose(self):
        A = self.S.transpose()
        np.testing.assert_array_almost_equal(self.P.T, A.compute_product())

    def test_compute_spectral_norm(self):
        np.testing.assert_almost_equal(
            self.S.compute_spectral_norm(method='eigs')[0],
            np.linalg.norm(self.P, ord=2))
        np.testing.assert_almost_equal(
            self.S.compute_spectral_norm(method='svds')[0],
            np.linalg.norm(self.P, ord=2))

    def test_io(self):
        np.save('tmp.npy', self.S, allow_pickle=True)
        S_loaded = np.load('tmp.npy', allow_pickle=True)
        print(self.S)
        print(S_loaded)
        # TODO test equality

    def test_apply_L(self):
        for n_factors in range(self.S.n_factors + 1):
            if n_factors < self.S.n_factors:
                d = np.min(self.S.shape)
            else:
                d = self.S.shape[1]
            X_vec = np.random.randn(d)
            X_mat = np.random.randn(d, 10)
            for X in [X_vec, X_mat]:
                if n_factors == 0:
                    y_ref = X
                elif n_factors == 1:
                    y_ref = self.A[0] @ X
                else:
                    y_ref = np.linalg.multi_dot(self.A[:n_factors]) @ X
                y_est = self.S.apply_L(n_factors=n_factors, X=X)
                np.testing.assert_array_almost_equal(
                    y_est,
                    y_ref,
                    err_msg='{} factors, {} data'.format(n_factors, X.shape))

    def test_apply_LH(self):
        for n_factors in range(self.S.n_factors + 1):
            d = self.S.shape[0]
            X_vec = np.random.randn(d)
            X_mat = np.random.randn(d, 10)
            for X in [X_vec, X_mat]:
                if n_factors == 0:
                    y_ref = X
                elif n_factors == 1:
                    y_ref = np.conjugate(self.A[0]).T @ X
                else:
                    L = np.linalg.multi_dot(self.A[:n_factors])
                    y_ref = np.conjugate(L.T) @ X
                y_est = self.S.apply_LH(n_factors=n_factors, X=X)
                np.testing.assert_array_almost_equal(
                    y_est,
                    y_ref,
                    err_msg='{} factors, {} data'.format(n_factors, X.shape))

    def test_apply_R(self):
        for n_factors in range(self.S.n_factors + 1):
            if n_factors < self.S.n_factors:
                d = np.min(self.S.shape)
            else:
                d = self.S.shape[0]
            X_vec = np.random.randn(d)
            X_mat = np.random.randn(10, d)
            for X in [X_vec, X_mat]:
                if n_factors == 0:
                    y_ref = X
                elif n_factors == 1:
                    y_ref = X @ self.A[-1]
                else:
                    y_ref = X @ np.linalg.multi_dot(self.A[-n_factors:])
                y_est = self.S.apply_R(n_factors=n_factors, X=X)
                np.testing.assert_array_almost_equal(
                    y_est,
                    y_ref,
                    err_msg='{} factors, {} data'.format(n_factors, X.shape))

    def test_apply_RH(self):
        for n_factors in range(self.S.n_factors + 1):
            d = self.S.shape[1]
            X_vec = np.random.randn(d)
            X_mat = np.random.randn(10, d)
            for X in [X_vec, X_mat]:
                if n_factors == 0:
                    y_ref = X
                elif n_factors == 1:
                    y_ref = X @ np.conjugate(self.A[-1].T)
                else:
                    R = np.linalg.multi_dot(self.A[-n_factors:])
                    y_ref = X @ np.conjugate(R.T)
                y_est = self.S.apply_RH(n_factors=n_factors, X=X)
                np.testing.assert_array_almost_equal(
                    y_est,
                    y_ref,
                    err_msg='{} factors, {} data'.format(n_factors, X.shape))


if __name__ == '__main__':
    unittest.main()
