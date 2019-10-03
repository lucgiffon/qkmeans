"""
Kernel related functions.
"""

import numpy as np
from qkmeans.data_structures import SparseFactors
from qkmeans.utils import logger
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms


def special_rbf_kernel(X, Y, gamma, norm_X=None, norm_Y=None, exp_outside=True):
    """
    Rbf kernel expressed under the form f(x)f(u)f(xy^T)

    Can handle X and Y as Sparse Factors.

    :param X: n x d matrix
    :param Y: n x d matrix
    :param gamma:
    :param norm_X: nx1 matrix
    :param norm_Y: 1xn matrix
    :param exp_outside: Tells if the exponential should be computed just once. Numerical instability may arise if False.
    :return:
    """
    assert len(X.shape) == len(Y.shape) == 2

    if norm_X is None:
        norm_X = row_norms(X, squared=True)[:, np.newaxis]
    else:
        norm_X = check_array(norm_X)

    if norm_Y is None:
        norm_Y = row_norms(Y, squared=True)[np.newaxis, :]
    else:
        norm_Y = check_array(norm_Y)

    def f(norm_mat):
        return np.exp(-gamma * norm_mat)

    def g(scal):
        return np.exp(2 * gamma * scal)

    if isinstance(X, SparseFactors) and isinstance(Y, SparseFactors):
        # xyt = SparseFactors(X.get_list_of_factors() + Y.transpose().get_list_of_factors()).compute_product(return_array=True)
        S = SparseFactors(lst_factors=X.get_list_of_factors() + Y.get_list_of_factors_H(), lst_factors_H=X.get_list_of_factors_H() + Y.get_list_of_factors())
        xyt = S.compute_product(return_array=True)
    elif not isinstance(X, SparseFactors) and isinstance(Y, SparseFactors):
        xyt = (Y @ X.transpose()).transpose()
    else:
        xyt = X @ Y.transpose()

    if not exp_outside:
        return f(norm_X) * g(xyt) * f(norm_Y)
    else:
        distance = -2 * xyt
        distance += norm_X
        distance += norm_Y
        # distance = norm_X + norm_Y - (2 * xyt)
        np.maximum(distance, 0, out=distance)
        if X is Y:
            np.fill_diagonal(distance, 0)

        in_exp = -gamma * distance
        return np.exp(in_exp)

def prepare_nystrom(landmarks, landmarks_norm, gamma):
    """
    Return the K^{-1/2} matrix of Nyström: the metric used for the transformation.

    It uses the rbf kernel.

    :param landmarks: The matrix of landmark points
    :param landmarks_norm: The norm of the matrix of landmark points
    :param gamma: The gamma value to use in the rbf kernel.
    :return:
    """
    basis_kernel_W = special_rbf_kernel(landmarks, landmarks, gamma, landmarks_norm, landmarks_norm.T)
    U, S, V = np.linalg.svd(basis_kernel_W)
    Sprim =  np.maximum(S, 1e-12)
    if (Sprim != S).any():
        logger.warning("One value of S in singular decomposition of W was lower than 1e-12")
    S = Sprim

    normalization_ = np.dot(U / np.sqrt(S), V)

    return normalization_

def nystrom_transformation(x_input, landmarks, p_metric, landmarks_norm, x_input_norm, gamma):
    """
    Apply the nystrom transformation given the metric.

    It uses the rbf kernel.

    :param x_input: The inputs to transform.
    :param landmarks: The landmark points used in the nystrom transformation.
    :param p_metric: The metric used by the transformation (usually K^{-1/2})
    :param landmarks_norm: The norm of the landmark points
    :param x_input_norm: The norm of the input.
    :param gamma: The gamma value for the rbf kernel.
    :return:
    """


    nystrom_embedding = special_rbf_kernel(landmarks, x_input, gamma, landmarks_norm, x_input_norm).T @ p_metric
    return nystrom_embedding