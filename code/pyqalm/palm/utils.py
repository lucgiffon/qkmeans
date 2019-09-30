import numpy as np
from numpy.linalg import multi_dot
from pyqalm.data_structures import SparseFactors


def compute_objective_function(arr_X_target, _f_lambda, _lst_S):
    if isinstance(_lst_S, SparseFactors):
        reconstruct = _f_lambda * _lst_S.compute_product()
    else:
        reconstruct = _f_lambda * multi_dot(_lst_S)
    return np.linalg.norm(arr_X_target - reconstruct, ord='fro') ** 2

def update_scaling_factor(X, X_est):
    return np.sum(X * X_est) / np.sum(X_est ** 2)