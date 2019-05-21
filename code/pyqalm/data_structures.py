# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, svds


class SparseFactors(LinearOperator):
    def __init__(self, lst_factors=[]):
        self._lst_factors = [csr_matrix(x) for x in lst_factors]
        for i in range(len(lst_factors)):
            assert self._lst_factors[i].ndim == 2
        for i in range(len(lst_factors) - 1):
            assert self._lst_factors[i].shape[1] \
                   == self._lst_factors[i + 1].shape[0]

    @property
    def n_factors(self):
        return len(self._lst_factors)

    @property
    def shape(self):
        if len(self._lst_factors) == 0:
            return 0, 0
        return self._lst_factors[0].shape[0], self._lst_factors[-1].shape[-1]

    @property
    def dtype(self):
        """

        Returns
        -------

        """
        return np.dtype(np.prod([x[0, 0] for x in self._lst_factors]))

    def set_factor(self, index, x):
        self._lst_factors[index] = csr_matrix(x)
        if index < 0:
            index += self.n_factors
        if index > 0:
            assert self._lst_factors[index].shape[0] \
                   == self._lst_factors[index - 1].shape[1]
        if index < self.n_factors - 1:
            assert self._lst_factors[index].shape[1] \
                   == self._lst_factors[index + 1].shape[0]

    def __call__(self, x):
        """
        Call self as a function.

        Parameters
        ----------
        x

        Returns
        -------

        """
        return self.dot(x)

    def _adjoint(self):
        """
        Hermitian adjoint.

        Returns
        -------

        """
        return SparseFactors([x.getH() for x in reversed(self._lst_factors)])

    def _matmat(self, X):
        """
        Matrix-matrix multiplication.

        Parameters
        ----------
        X

        Returns
        -------

        """
        for a in reversed(self._lst_factors):
            X = a.dot(X)
        return X

    def transpose(self):
        """
        Transpose this linear operator.

        Returns
        -------

        """

        return SparseFactors([x.transpose()
                              for x in reversed(self._lst_factors)])

    def compute_product(self):
        Y = self._lst_factors[-1]
        for X in reversed(self._lst_factors[:-1]):
            Y = X.dot(Y)
        return Y.toarray()

    def get_factor(self, index, copy=False):
        if copy:
            return self._lst_factors[index].copy()
        else:
            return self._lst_factors[index]

    def get_list_of_factors(self, copy=False):
        if copy:
            return [x.copy() for x in self._lst_factors]
        else:
            return self._lst_factors

    def compute_spectral_norm(self, method='eigs'):
        if method == 'svds':
            a = svds(A=self, k=1, return_singular_vectors=False)
            return a[0]
        elif method == 'eigs':
            if self.shape[0] > self.shape[1]:
                SS = SparseFactors(self.adjoint().get_list_of_factors()
                                   + self.get_list_of_factors())
            else:
                SS = SparseFactors(self.get_list_of_factors()
                                   + self.adjoint().get_list_of_factors())
            a = eigs(A=SS, k=1, return_eigenvectors=False)
            return np.sqrt(np.real(a[0]))


if __name__ == '__main__':
    import numpy as np
    A = [np.random.randint(0, 3, size=(4, 4)) for _ in range(2)]
    A += [np.random.randint(0, 3, size=(4, 5))]
    P = np.linalg.multi_dot(A)
    print(A)
    O = SparseFactors(A)
    print(O.dtype)
    print(O.adjoint())
    x = np.random.randn(O.shape[1])[:, None]
    print(P @ x - O @ x)
    print(P - O.compute_product())
    print(P.T - O.transpose().compute_product())
    print(np.conjugate(P).T-O.adjoint().get_product())
