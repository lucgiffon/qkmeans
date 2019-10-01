import numpy as np
from qkmeans.data_structures import SparseFactors
from qkmeans.core.utils import get_squared_froebenius_norm_line_wise
from qkmeans.utils import logger


def special_rbf_kernel(X, Y, gamma, norm_X, norm_Y, exp_outside=True):
    """
    Rbf kernel expressed under the form f(x)f(u)f(xy^T)

    Can handle X and Y as Sparse Factors.

    :param X: n x d matrix
    :param Y: n x d matrix
    :return:
    """
    assert len(X.shape) == len(Y.shape) == 2

    if norm_X is None:
        norm_X = get_squared_froebenius_norm_line_wise(X)
    if norm_Y is None:
        norm_Y = get_squared_froebenius_norm_line_wise(Y)

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
        return f(norm_X).reshape(-1, 1) * g(xyt) * f(norm_Y).reshape(1, -1)
    else:
        distance = (norm_X.reshape(-1, 1)) + (norm_Y.reshape(1, -1)) - (2 * xyt)
        np.maximum(distance, 0, out=distance)
        if X is Y:
            np.fill_diagonal(distance, 0)
        in_exp = -gamma * distance
        return np.exp(in_exp)

def prepare_nystrom(landmarks, landmarks_norm, gamma):
    basis_kernel_W = special_rbf_kernel(landmarks, landmarks, gamma, landmarks_norm, landmarks_norm)
    U, S, V = np.linalg.svd(basis_kernel_W)
    Sprim =  np.maximum(S, 1e-12)
    if (Sprim != S).any():
        logger.warning("One value of S in singular decomposition of W was lower than 1e-12")
    S = Sprim

    normalization_ = np.dot(U / np.sqrt(S), V)

    return normalization_

def nystrom_transformation(x_input, landmarks, p_metric, landmarks_norm, x_input_norm, gamma):
    nystrom_embedding = special_rbf_kernel(landmarks, x_input, gamma, landmarks_norm, x_input_norm).T @ p_metric
    return nystrom_embedding