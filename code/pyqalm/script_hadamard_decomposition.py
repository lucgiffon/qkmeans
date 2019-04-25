# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon
"""
import numpy as np
from scipy.linalg import hadamard
from numpy.linalg import norm

from pyqalm.qalm import HierarchicalPALM4MSA
from pyqalm.test_qalm import visual_evaluation_palm4msa


d = 32
nb_iter = 5000
nb_factors = 5
nb_keep_values = 64

# lst_factors = [projection_operator(np.random.rand(d, d), nb_keep_values) for _ in range(nb_factors)]
lst_factors = [np.eye(d) for _ in range(nb_factors)]
#lst_factors = [np.random.rand(d, d) for _ in range(nb_factors)]
lst_factors = [fac/norm(fac) for fac in lst_factors]
lst_factors[-1] = np.zeros((d, d))  # VE
_lambda = 1.
had = hadamard(d)
H =  had / norm(had, ord='fro')
print(H)

#final_lambda, final_factors, final_X = PALM4LED(H, lst_factors, [nb_keep_values for _ in range(nb_factors)], _lambda, nb_iter)
final_lambda, final_factors, final_X = HierarchicalPALM4MSA(H, lst_factors, nb_keep_values, _lambda, nb_iter, right_to_left=True)

print("Lambda value: " + str(final_lambda))
visual_evaluation_palm4msa(H, lst_factors, final_factors, final_X)