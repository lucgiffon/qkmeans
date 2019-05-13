# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon
"""
import logging
import daiquiri

import numpy as np
from pyqalm.utils import logger
from scipy.linalg import hadamard
from numpy.linalg import norm

from pyqalm.qalm import HierarchicalPALM4MSA
from pyqalm.test.test_qalm import visual_evaluation_palm4msa


daiquiri.setup(level=logging.DEBUG)
d = 32
nb_iter = 30
nb_factors = 5
nb_keep_values = 64

# lst_factors = [projection_operator(np.random.rand(d, d), nb_keep_values) for _ in range(nb_factors)]
lst_factors = [np.eye(d) for _ in range(nb_factors)]
#lst_factors = [np.random.rand(d, d) for _ in range(nb_factors)]
# lst_factors = [fac/norm(fac) for fac in lst_factors]
lst_factors[-1] = np.zeros((d, d))  # VE
_lambda = 1.
had = hadamard(d)
#H =  had / norm(had, ord='fro')
# H = had
H = had / np.sqrt(32)

#final_lambda, final_factors, final_X = PALM4LED(H, lst_factors, [nb_keep_values for _ in range(nb_factors)], _lambda, nb_iter)
final_lambda, final_factors, final_X = HierarchicalPALM4MSA(
    arr_X_target=H,
    lst_S_init=lst_factors,
    nb_keep_values=nb_keep_values,
    f_lambda_init=_lambda,
    nb_iter=nb_iter,
    right_to_left=True)

visual_evaluation_palm4msa(H, lst_factors, final_factors, final_X)

vec = np.random.rand(d)
h_vec = H @ vec
r_vec = final_X @ vec
logger.debug("Distance matrice to random vector (true vs fake):{}".format(norm(h_vec - r_vec)))