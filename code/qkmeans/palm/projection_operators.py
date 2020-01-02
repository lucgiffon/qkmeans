"""
Matrix-projection related functions.
"""

import numpy as np
from numpy import argpartition
from numpy.matlib import repmat


def get_unraveled_indexes_from_index_array(index_arr, shape, order='C'):
    """
    Inplace rearrange or array values following index_arr.

    :param array:
    :param index_arr:
    :return:
    """
    incre = np.arange(0, shape[0] * shape[1] - 1, shape[0])  # the vector of idx of the first cell of each column (in the flattened vector)
    incremat = repmat(incre, index_arr.shape[0], 1)
    index_arr = index_arr + incremat

    index_arr = np.ravel(index_arr)

    return np.unravel_index(
        index_arr,
        shape,
        order)


def projection_operator(input_arr, nb_keep_values):
    """
    Project input_arr onto its nb_keep_values highest values
    """
    flat_input_arr = input_arr.flatten()
    # find the index of the lowest values (not the _nb_keep_values highest)
    lowest_values_idx = argpartition(np.absolute(flat_input_arr), -nb_keep_values, axis=None)[:-nb_keep_values]
    # set the value of the lowest values to zero
    flat_input_arr[lowest_values_idx] = 0.
    # return reshape_to_matrix
    return flat_input_arr.reshape(input_arr.shape)


def inplace_hardthreshold(input_arr, nb_keep_values):
    """
    Hard-threshold input_arr by keeping its nb_keep_values highest values only
    Variant without copy of input_arr (inplace changes)
    """
    # find the index of the lowest values (not the _nb_keep_values highest)
    lowest_values_idx = argpartition(np.absolute(input_arr), -nb_keep_values, axis=None)[:-nb_keep_values]
    # set the value of the lowest values to zero
    input_arr.reshape(-1)[lowest_values_idx] = 0


def _projection_max_by_col(X, nb_val_by_col, fast_unstable):
    """
    Return a copy of `X` but with only the `nb_val_by_col` biggest (in amplitude) values in each columns (other values are zero-ed)

    Parameters
    ----------
    X
    nb_val_by_col
    fast_unstable

    Returns
    -------

    """
    nb_val_by_col = round(nb_val_by_col)
    Xabs = np.abs(X)  # biggest amplitude
    Xprox_col = np.zeros_like(X)
    if not fast_unstable:
        sortIndex = np.argsort(-Xabs, axis=0,
                               kind="stable")  # -Xabs for sort in descending order
        maxIndex = sortIndex[:nb_val_by_col, :]
    else:
        maxIndex = np.argpartition(-Xabs, nb_val_by_col, axis=0)[
                   :nb_val_by_col, :]
        # todo peut-être faire un tri après pour plus de stabilité

    unraveled_indices = get_unraveled_indexes_from_index_array(maxIndex,
                                                               shape=X.shape,
                                                               order="F")

    Xprox_col[unraveled_indices] = X[unraveled_indices]
    return Xprox_col


def prox_splincol(input_arr, nb_val_total, fast_unstable=False):
    input_arr = np.round(input_arr, 10)  # maybe use hard decimal cut ? I don't know

    fraction_by_row_col = input_arr.shape[0] * input_arr.shape[1] / nb_val_total
    nb_val_by_col = int(input_arr.shape[0] / fraction_by_row_col)
    nb_val_by_row = int(input_arr.shape[1] / fraction_by_row_col)

    Xprox_col = _projection_max_by_col(input_arr, nb_val_by_col, fast_unstable)
    Xprox_lin = _projection_max_by_col(input_arr.T, nb_val_by_row, fast_unstable).T
    Xprox = Xprox_col + Xprox_lin * (Xprox_col == 0)

    return Xprox