# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon
"""
import logging
import daiquiri
import pprint
import numpy as np
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.utils import logger
from scipy.linalg import hadamard
from numpy.linalg import norm

from qkmeans.palm.palm_fast import hierarchical_palm4msa
from qkmeans.utils import visual_evaluation_palm4msa

daiquiri.setup(level=logging.INFO)

# Create matrix to approximate: hadamard matrix
d = 32
H = hadamard(d)

# Parameters for palm
nb_iter = 300
nb_factors = 5
sparsity_factor = 2

# Create init sparse factors as identity (the first sparse matrix will remain constant)
lst_factors = [np.eye(d) for _ in range(nb_factors + 1)]
lst_factors[-1] = np.zeros((d, d))
_lambda = 1.  # init the scaling factor at 1

# Create the projection operators for each factor
lst_proj_op_by_fac_step, lst_proj_op_by_fac_step_desc = build_constraint_set_smart(left_dim=d,
                                                     right_dim=d,
                                                     nb_factors=nb_factors + 1,
                                                     sparsity_factor=sparsity_factor,
                                                     residual_on_right=True,
                                                     fast_unstable_proj=False)

logger.info("Description of projection operators for each iteration of hierarchical_palm: \n{}".format(pprint.pformat(lst_proj_op_by_fac_step_desc)))

# Call the algorithm
final_lambda, final_factors, final_X, _, _ = hierarchical_palm4msa(
    arr_X_target=H,
    lst_S_init=lst_factors,
    lst_dct_projection_function=lst_proj_op_by_fac_step,
    f_lambda_init=_lambda,
    nb_iter=nb_iter,
    update_right_to_left=True,
    residual_on_right=True)

# Vizualization utility
visual_evaluation_palm4msa(H, lst_factors, final_factors, final_X)

vec = np.random.rand(d)
h_vec = H @ vec
r_vec = final_X @ vec
logger.info("Distance matrice to random vector (true vs fake):{}".format(norm(h_vec - r_vec)/np.linalg.norm(r_vec)))