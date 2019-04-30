import unittest
import numpy as np
import matplotlib.pyplot as plt

from pyqalm.qalm import projection_operator, get_side_prod, \
    inplace_hardthreshold, update_scaling_factor


def test_get_side_prod():
    # TODO refactor into a unittest
    nb_factors = 3
    d = 32
    nb_keep_values =64
    factors = [projection_operator(np.random.rand(d, d), nb_keep_values) for _ in range(nb_factors)]
    result = get_side_prod(factors)
    # truth =
    visual_evaluation_palm4msa()


def test_projection_operator():
    # TODO refactor into a unittest
    matrix = np.random.permutation(16).reshape(4, 4) - 8
    print(matrix)
    matrix_proj = projection_operator(matrix, 5)
    print(matrix_proj)

test_projection_operator()


def test_inplace_hardthreshold():
    # TODO refactor into a unittest
    matrix = np.random.permutation(16).reshape(4, 4) -8
    print(matrix)
    inplace_hardthreshold(matrix, 5)
    print(matrix)

test_inplace_hardthreshold()



def visual_evaluation_palm4msa(target, init_factors, final_factors, result):
    nb_factors = len(init_factors)
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 2, 1)
    plt.imshow(target)
    plt.subplot(3, 2, 2)
    plt.imshow(result)
    print("Première ligne: Objectif \t | \t Résultat")
    print("Deuxième ligne: Les facteurs")
    print("Troisième ligne: Les facteurs initiaux")
    for i in range(nb_factors):
        plt.subplot(3, nb_factors, nb_factors + (i+1))
        plt.imshow(final_factors[i])
        plt.subplot(3, nb_factors, nb_factors + nb_factors + (i+1))
        plt.imshow(init_factors[i])


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

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
