# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
import numpy as np


class LinearOperator:
    def __init__(self, matrix):
        assert isinstance(matrix, np.ndarray)
        self._dense_matrix = matrix

    @property
    def row_square_norms(self):
        return np.sum(self._dense_matrix**2, axis=1)

    def __call__(self, x):
        """ (Left-)Apply the linear operator to a vector or a matrix. """
        return self._dense_matrix @ x

    def assign(self, X):
        ux = self(X.T)
        u2 = self.row_square_norms
        return np.argmin(u2[:, None] - 2 * ux, axis=0)


class FastLinearOperator(LinearOperator):
    def __init__(self, factors):
        self.factors = factors

    @property
    def _dense_matrix(self):
        M = self.factors[-1]
        for i in range(self.n_factors-1):
            M = self.factors[-i-1].dot(M)
        return M.toarray()

    def __call__(self, x):
        for i in range(self.n_factors):
            x = self.factors[-i-1].dot(x)
        return x


if __name__ == '__main__':
    from scipy.linalg import hadamard as sp_hadamard_matrix
    transform_length = 17
    n_examples = 15
    d = 13
    # M = sp_hadamard_matrix(d)
    M = np.random.randn(transform_length, d)
    X = np.random.randn(n_examples, d)

    linop = LinearOperator(M)

    # test row norms
    row_norms = linop.row_square_norms
    assert row_norms.size == transform_length
    print(np.max(np.abs(row_norms - np.linalg.norm(M, axis=1)**2)))

    # test call
    print(np.max(np.abs(linop(X.T) - M@X.T)))

    # test assign
    t = linop.assign(X)
    print(t)
    print(np.all(t < transform_length))
