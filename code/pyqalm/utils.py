import logging

from numpy import identity
from numpy.linalg import multi_dot
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger("pyqalm")

def get_side_prod(lst_factors, id_size=0):
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
        side_prod = identity(id_size)
    elif len(lst_factors) == 1:
        # if only 1 elm, return the elm itself (Identity * elm actually)
        side_prod = lst_factors[0]
    else:
        side_prod = multi_dot(lst_factors)
    return side_prod