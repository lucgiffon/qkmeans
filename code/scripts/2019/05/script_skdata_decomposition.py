# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon
"""
import logging
from pprint import pformat

import daiquiri
import numpy as np
from qkmeans.utils import logger, get_lambda_proxsplincol
from numpy.linalg import norm
from sklearn import datasets

from qkmeans.palm.palm import hierarchical_palm4msa
from qkmeans.utils import visual_evaluation_palm4msa

daiquiri.setup(level=logging.INFO)

# iris = datasets.load_iris(); X = iris.data
# boston = datasets.load_boston(); X = boston.data
X, _ = datasets.make_blobs(n_samples=200, n_features=20, centers=4)

nb_iter = 500
residual_on_right = True
nb_factors = 10
d = min(X.shape)

lst_factors = [np.eye(min(X.shape)) for _ in range(nb_factors)]
lst_factors[-1] = np.random.rand(min(X.shape), X.shape[1])
lst_factors[0] = np.eye(X.shape[0], min(X.shape))
_lambda = 1.

lst_proj_op_by_fac_step = []
factor = 10
nb_keep_values = factor*d
for k in range(nb_factors - 1):
    nb_values_residual = max(nb_keep_values, int(d / 2 ** (k + 1)) * d)
    dct_step_lst_nb_keep_values = {
        "split": [get_lambda_proxsplincol(nb_keep_values), get_lambda_proxsplincol(nb_values_residual)] if residual_on_right else [get_lambda_proxsplincol(nb_values_residual), get_lambda_proxsplincol(nb_keep_values)],
        "finetune": [get_lambda_proxsplincol(nb_keep_values)] * (k+1) + [get_lambda_proxsplincol(nb_values_residual)] if residual_on_right else [get_lambda_proxsplincol(nb_values_residual)] + [get_lambda_proxsplincol(nb_keep_values)] * (k+1)
    }
    lst_proj_op_by_fac_step.append(dct_step_lst_nb_keep_values)

logger.info("Sparsity parameter by factor: {}".format(pformat(lst_proj_op_by_fac_step)))
#final_lambda, final_factors, final_X = PALM4LED(H, lst_factors, [nb_keep_values for _ in range(nb_factors)], _lambda, nb_iter)
final_lambda, final_factors, final_X, nb_iter_by_factor, _ = hierarchical_palm4msa(
    arr_X_target=X,
    lst_S_init=lst_factors,
    lst_dct_projection_function=lst_proj_op_by_fac_step,
    f_lambda_init=_lambda,
    nb_iter=nb_iter,
    update_right_to_left=True,
    residual_on_right=residual_on_right,
    graphical_display=False)

logger.info("Number of iteration for each factor: {}; Total: {}".format(nb_iter_by_factor, sum(nb_iter_by_factor)))

visual_evaluation_palm4msa(X, lst_factors, final_factors, final_X)

normalized_diff = final_X/norm(final_X) - X/norm(X)

nb_vec = 100
lst_random_vec = [np.random.rand(X.shape[1]) for _ in range(nb_vec)]
lst_operator_res_obj = [X @ v for v in lst_random_vec]
lst_operator_res_final = [final_X @ v for v in lst_random_vec]
lst_diff_operator = [np.linalg.norm(lst_operator_res_obj[i]/np.linalg.norm(lst_operator_res_obj[i]) - lst_operator_res_final[i]/np.linalg.norm(lst_operator_res_final[i])) for i in range(nb_vec)]
logger.info("Normalized difference between matrices: {}".format(np.linalg.norm(normalized_diff)))
logger.info("Biggest delta: {}".format(normalized_diff.max()))
logger.info("Mean operator difference: {}".format(np.mean(lst_diff_operator)))