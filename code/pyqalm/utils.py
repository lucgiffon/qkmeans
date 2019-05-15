import logging

from numpy import eye
from numpy.linalg import multi_dot
import daiquiri
from pyqalm.projection_operators import prox_splincol

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger("pyqalm")

def get_side_prod(lst_factors, id_shape=(0,0)):
    """
    Return the dot product between factors in lst_factors in order.

    exemple:

    lst_factors := [S1, S2, S3]
    return_value:= S1 @ S2 @ S3
    """
    # assert if the inner dimension of factors match: e.g. the multi dot product is feasible
    assert all([lst_factors[i].shape[-1] == lst_factors[i+1].shape[0] for i in range(len(lst_factors)-1)])

    if len(lst_factors) == 0:
        # convention from the paper itself: dot product of no factors equal Identity
        side_prod = eye(*id_shape)
    elif len(lst_factors) == 1:
        # if only 1 elm, return the elm itself (Identity * elm actually)
        side_prod = lst_factors[0]
    else:
        side_prod = multi_dot(lst_factors)
    return side_prod

def get_lambda_proxsplincol(nb_keep_values):
    return lambda mat: prox_splincol(mat, nb_keep_values)

def constant_proj(mat):
    raise NotImplementedError("This function should not be called but used for its name")