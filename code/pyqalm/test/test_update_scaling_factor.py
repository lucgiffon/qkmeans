import unittest
import numpy as np
from pyqalm.palm.utils import update_scaling_factor


class TestUpdateScalingFactor(unittest.TestCase):
    def test_equality_with_trace_computation(self):
        X_shape = 123, 234
        X = np.random.randn(*X_shape)
        X_est = np.random.randn(*X_shape)
        actual = update_scaling_factor(X=X, X_est=X_est)
        desired = np.trace(X.T @ X_est) / np.trace(X_est.T @ X_est)
        np.testing.assert_almost_equal(actual=actual, desired=desired)

    def test_basic_cases(self):
        X_shape = 123, 234
        X = np.random.randn(*X_shape)
        actual = update_scaling_factor(X=X, X_est=X)
        desired = 1
        np.testing.assert_almost_equal(actual=actual, desired=desired)


if __name__ == '__main__':
    unittest.main()
