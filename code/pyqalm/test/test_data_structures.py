import unittest
import numpy as np

from pyqalm.data_structures import SparseFactors


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
            self.S.compute_spectral_norm(method='eigs'),
            np.linalg.norm(self.P, ord=2))
        np.testing.assert_almost_equal(
            self.S.compute_spectral_norm(method='svds'),
            np.linalg.norm(self.P, ord=2))

    def test_io(self):
        np.save('tmp.npy', self.S, allow_pickle=True)
        S_loaded = np.load('tmp.npy', allow_pickle=True)
        print(self.S)
        print(S_loaded)
        # TODO test equality

if __name__ == '__main__':
    unittest.main()
