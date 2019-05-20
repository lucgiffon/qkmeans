# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon
"""
import logging
import daiquiri

import numpy as np
from pyqalm.utils import logger, get_lambda_proxsplincol
from scipy.linalg import hadamard
from numpy.linalg import norm

from pyqalm.qalm import HierarchicalPALM4MSA
from pyqalm.test.test_qalm import visual_evaluation_palm4msa


daiquiri.setup(level=logging.DEBUG)
d = 32
nb_iter = 300
nb_factors = 5


lst_factors = [np.eye(d) for _ in range(nb_factors)]
lst_factors[-1] = np.zeros((d, d))  # VE
_lambda = 1.
had = hadamard(d)
#H =  had / norm(had, ord='fro')
H = had / np.sqrt(32)

lst_proj_op_by_fac_step = []
nb_keep_values = 2*d
for k in range(nb_factors - 1):
    nb_values_residual = int(d / 2 ** (k + 1)) * d
    dct_step_lst_nb_keep_values = {
        "split": [get_lambda_proxsplincol(nb_keep_values), get_lambda_proxsplincol(nb_values_residual)],
        "finetune": [get_lambda_proxsplincol(nb_keep_values)] * (k+1) + [get_lambda_proxsplincol(nb_values_residual)]
    }
    lst_proj_op_by_fac_step.append(dct_step_lst_nb_keep_values)

#final_lambda, final_factors, final_X = PALM4LED(H, lst_factors, [nb_keep_values for _ in range(nb_factors)], _lambda, nb_iter)
final_lambda, final_factors, final_X, nb_iter_by_factor, _ = HierarchicalPALM4MSA(
    arr_X_target=H,
    lst_S_init=lst_factors,
    lst_dct_projection_function=lst_proj_op_by_fac_step,
    f_lambda_init=_lambda,
    nb_iter=nb_iter,
    update_right_to_left=True,
    residual_on_right=True,
    graphical_display=False)

logger.debug("Number of iteration for each factor: {}; Total: {}".format(nb_iter_by_factor, sum(nb_iter_by_factor)))

visual_evaluation_palm4msa(H, lst_factors, final_factors, final_X)

vec = np.random.rand(d)
h_vec = H @ vec
r_vec = final_X @ vec
logger.debug("Distance matrice to random vector (true vs fake):{}".format(norm(h_vec - r_vec)))