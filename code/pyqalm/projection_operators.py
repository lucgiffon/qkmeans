import numpy as np
from numpy import argpartition
from numpy.matlib import repmat


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


def prox_splincol(input_arr, nb_val_total):
    def projection_max_by_col(X, nb_val_by_col):
        nb_val_by_col = round(nb_val_by_col)
        Xabs = np.abs(X)
        Xprox_col = np.zeros_like(X)
        sortIndex = np.argsort(-Xabs, axis=0, kind="stable") # -Xabs for sort in descending order
        maxIndex = sortIndex[:nb_val_by_col, :]
        incre = np.arange(0, X.shape[0] * X.shape[1]-1, X.shape[0]) # the vector of idx of the first cell of each column (in the flattened vector)
        incremat = repmat(incre, nb_val_by_col, 1)
        maxIndex = maxIndex + incremat # type: np.ndarray
        maxIndex = maxIndex.flatten() # index of the column-wise maximum values (in the flattened version of the input array)
        unraveled_indices = np.unravel_index(maxIndex, Xabs.shape, order='F') # order=F: translation from matlab code with column major indexing (Fortran style)
        Xprox_col[unraveled_indices] = X[unraveled_indices]
        return Xprox_col

    input_arr = np.round(input_arr, 10) # maybe use hard decimal cut ? I don't know

    fraction_by_row_col = input_arr.shape[0] * input_arr.shape[1] / nb_val_total
    nb_val_by_col = int(input_arr.shape[0] / fraction_by_row_col)
    nb_val_by_row = int(input_arr.shape[1] / fraction_by_row_col)

    Xprox_col = projection_max_by_col(input_arr, nb_val_by_col)
    Xprox_lin = projection_max_by_col(input_arr.T, nb_val_by_row).T
    Xprox = Xprox_col + Xprox_lin * (Xprox_col == 0)

    return Xprox