import numpy as np
from numpy.linalg import multi_dot
from qkmeans.data_structures import SparseFactors


def compute_objective_function(arr_X_target, _f_lambda, _lst_S):
    """
    Compute objective function for the palm4msa algorithm: the froebenius norm of the difference between the target matrix and the reconstructed factorization.

    :param arr_X_target: The target matrix
    :param _f_lambda: The scaling factor
    :param _lst_S: The sparse factors
    :return:
    """
    if isinstance(_lst_S, SparseFactors):
        reconstruct = _f_lambda * _lst_S.compute_product()
    else:
        reconstruct = _f_lambda * multi_dot(_lst_S)
    return np.linalg.norm(arr_X_target - reconstruct, ord='fro') ** 2

def update_scaling_factor(X, X_est):
    """
    Implementation of the scaling factor between the reconstructed matrix and the target matrix

    :param X: Reconstructed matrix from unit norm sparse factors
    :param X_est: Target matrix
    :return: The new scaling factor
    """
    return np.sum(X * X_est) / np.sum(X_est ** 2)