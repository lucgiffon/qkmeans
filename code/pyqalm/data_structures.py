# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, svds
from scipy.linalg import toeplitz


class SparseFactors(LinearOperator):
    # TODO include scaling factor into class SparseFactors
    def __init__(self, lst_factors=[], lst_factors_H=None):
        self._lst_factors = [x if isinstance(x, csr_matrix) else csr_matrix(x)
                             for x in lst_factors]
        # Maintain a list of Hermitian transpose factors
        if lst_factors_H is None:
            self._lst_factors_H = [x.getH()
                                   for x in reversed(self._lst_factors)]
        else:
            self._lst_factors_H = [x
                                   if isinstance(x, csr_matrix)
                                   else csr_matrix(x)
                                   for x in lst_factors_H]

        assert len(self._lst_factors) == len(self._lst_factors_H)
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

    def set_factor(self, index, x, xH=None):
        if not isinstance(x, csr_matrix):
            x = csr_matrix(x)
        if xH is None:
            xH = x.getH()
        elif not isinstance(xH, csr_matrix):
            xH = csr_matrix(xH)
        self._lst_factors[index] = x
        self._lst_factors_H[-index-1] = xH
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
        # return SparseFactors([x.getH() for x in reversed(self._lst_factors)])
        return SparseFactors(lst_factors=self._lst_factors_H,
                             lst_factors_H=self._lst_factors)

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

        # return SparseFactors([x.transpose()
        #                       for x in reversed(self._lst_factors)])
        return SparseFactors(lst_factors=[x.conjugate()
                                          for x in self._lst_factors_H],
                             lst_factors_H=[x.conjugate()
                                            for x in self._lst_factors])

    def compute_product(self, return_array=True):
        Y = self._lst_factors[-1]
        for X in reversed(self._lst_factors[:-1]):
            Y = X.dot(Y)
        if return_array:
            return Y.toarray()
        else:
            return Y

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

    def get_list_of_factors_H(self, copy=False):
        if copy:
            return [x.copy() for x in self._lst_factors_H]
        else:
            return self._lst_factors_H

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
            try:
                a = eigs(A=SS, k=1, return_eigenvectors=False)
            except Exception as e:
                warnings.warn(str(e))
                # FIXME if ARGPACK fails, compute norm with regular function
                return np.linalg.norm(self.compute_product(), ord=2)
                # return self.compute_spectral_norm(method='svds')
            return np.sqrt(np.real(a[0]))

    def get_L(self, n_factors):
        if n_factors == 0:
            return SparseFactors([], [])
        return SparseFactors(lst_factors=self._lst_factors[:n_factors],
                             lst_factors_H=self._lst_factors_H[-n_factors:])

    def get_R(self, n_factors):
        if n_factors == 0:
            return SparseFactors([], [])
        return SparseFactors(lst_factors=self._lst_factors[-n_factors:],
                             lst_factors_H=self._lst_factors_H[:n_factors])

    def apply_L(self, n_factors, X):
        """
        Apply several left factors to the left

        Parameters
        ----------
        n_factors : int
            Number of first left factors to be applied
        X : ndarray
            Vector or matrix

        Returns
        -------
        ndarray or csr_matrix
        """
        X_ndim = X.ndim
        if X_ndim == 1:
            X = X[:, None]
        for a in reversed(self._lst_factors[:n_factors]):
            X = a.dot(X)
        if X_ndim == 1:
            X = X.reshape(-1)
        return X

    def apply_LH(self, n_factors, X):
        """
        Apply adjoint of several left factors to the left

        Parameters
        ----------
        n_factors : int
            Number of first left factors to be applied
        X : ndarray
            Vector or matrix

        Returns
        -------
        ndarray or csr_matrix
        """
        if n_factors == 0:
            return X
        X_ndim = X.ndim
        if X_ndim == 1:
            X = X[:, None]
        # for a in self._lst_factors[:n_factors]:
        #     X = a.getH().dot(X)
        for a in reversed(self._lst_factors_H[-n_factors:]):
            X = a.dot(X)
        if X_ndim == 1:
            X = X.reshape(-1)
        return X

    def apply_R(self, n_factors, X):
        """
        Apply several right factors to the right

        Parameters
        ----------
        n_factors : int
            Number of last right factors to be applied
        X : ndarray
            Vector or matrix

        Returns
        -------
        ndarray or csr_matrix
        """
        if n_factors == 0:
            return X
        X_ndim = X.ndim
        if X_ndim == 1:
            X = X[:, None]
        else:
            X = X.T
        # for a in self._lst_factors[-n_factors:]:
        #     X = a.transpose().dot(X)
        for a in reversed(self._lst_factors_H[:n_factors]):
            X = a.conjugate().dot(X)
        if X_ndim == 1:
            return X.reshape(-1)
        else:
            return X.T

    def apply_RH(self, n_factors, X):
        """
        Apply adjoint of several right factors to the right

        Parameters
        ----------
        n_factors : int
            Number of last right factors to be applied
        X : ndarray
            Vector or matrix

        Returns
        -------
        ndarray or csr_matrix
        """
        if n_factors == 0:
            return X
        X_ndim = X.ndim
        if X_ndim == 1:
            X = X[:, None]
        else:
            X = X.T
        for a in reversed(self._lst_factors[-n_factors:]):
            X = a.conjugate().dot(X)
        if X_ndim == 1:
            return X.reshape(-1)
        else:
            return X.T


def permute_rows_cols_randomly(a):
    """
    Randomly permute the rows and columns of a matrix

    Parameters
    ----------
    a : np.ndarray [n, m]
        Matrix to be shuffled

    Returns
    -------
    np.ndarray [n, m]
        Shuffled matrix
    """
    # TODO testme
    assert a.ndim == 2
    return np.random.permutation(np.random.permutation(a).T).T


def create_factor_from_mask(mask):
    """
    Create sparse factors with random non-zero entries from a mask

    Parameters
    ----------
    mask : np.ndarray [n, m]
        Boolean mask where True values indicate the position of non-zero
        entries.

    Returns
    -------
    np.ndarray [n, m]
        Sparse matrix with non-zero entries drawn from a Gaussian distribution
    """
    # TODO testme
    A = np.zeros(mask.shape)
    A[mask] = np.random.randn(np.count_nonzero(mask))
    return A


def create_sparse_factors(axis_size, n_factors=None, sparsity_level=2):
    """
    Create sparse factors with a given sparsity level, created at random.

    Parameters
    ----------
    axis_size : int
        Number of rows and columns in sparse factors
    n_factors : int
        Number of factors. If None, set as the log2 of axis_size (rounded to
        the nearest upper integer).
    sparsity_level : int
        Number of non-zero values per row and column.

    Returns
    -------
    SparseFactors
        Sparse factors created at random.
    """
    # TODO testme
    if n_factors is None:
        n_factors = int(np.ceil(np.log2(axis_size)))
    c = np.array(
        [1] * sparsity_level + [0] * (axis_size - sparsity_level),
        dtype=bool)
    r = np.array(
        [1] + [0] * (axis_size - sparsity_level) + [1] * (sparsity_level - 1),
        dtype=bool)
    D = toeplitz(c, r)
    factors = [create_factor_from_mask(permute_rows_cols_randomly(D))
               for _ in range(n_factors)]
    S = SparseFactors(factors)
    return S
