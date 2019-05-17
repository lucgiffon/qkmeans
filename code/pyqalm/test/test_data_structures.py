import unittest
import numpy as np

from pyqalm.data_structures import SparseFactors


class TestSparseFactors(unittest.TestCase):
    # def setUpClass(cls) -> None:
    #     cls.A = [np.random.randint(0, 3, size=(4, 4)) for _ in range(2)]
    #     cls.A += [np.random.randint(0, 3, size=(4, 5))]
    #     cls.P = np.linalg.multi_dot(self.A)

    def setUp(self) -> None:
        self.A = [np.random.randint(0, 3, size=(4, 4)) for _ in range(2)]
        self.A += [np.random.randint(0, 3, size=(4, 5))]
        self.P = np.linalg.multi_dot(self.A)
        self.S = SparseFactors(self.A)

    def test_init_empty(self):
        S = SparseFactors([])
        np.testing.assert_array_equal(S.shape, np.array([]).shape)

    def test_shape(self):
        np.testing.assert_array_equal(
            self.S.shape,
            (self.A[0].shape[0], self.A[-1].shape[-1]))
        np.testing.assert_array_equal(self.S.shape, self.P.shape)

    def test_product(self):
        np.testing.assert_array_almost_equal(self.P,
                                             self.S.get_product().toarray())

    def test_matrix_vector_product(self):
        x = np.random.randn(self.S.shape[1])[:, None]
        Px = self.P @ x

        Sx = self.S.dot(x)
        np.testing.assert_array_almost_equal(Px, Sx)

        Sx = self.S @ x
        np.testing.assert_array_almost_equal(Px, Sx)

if __name__ == '__main__':
    unittest.main()
