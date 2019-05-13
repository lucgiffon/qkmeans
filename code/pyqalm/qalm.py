# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon
"""
from copy import deepcopy

import numpy as np
from numpy.linalg import norm
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt

from pyqalm.projection_operators import prox_splincol
from pyqalm.utils import get_side_prod, logger


def PALM4MSA(arr_X_target: np.array,
             lst_S_init: list,
             nb_factors: int,
             lst_nb_keep_values: list,
             f_lambda_init: float,
             nb_iter: int,
             update_right_to_left=True,
             graphical_display=False):
    """
    lst S init contains factors in decreasing indexes (e.g: the order along which they are multiplied in the product).
        example: S5 S4 S3 S2 S1

    lst S [-j] = Sj

    """

    def update_S(S_old, _left_side, _right_side, _c, _lambda, _nb_keep_values):
        """
        Return the new factor value.

        - Compute gradient
        - Do gradient step
        - Project data on _nb_keep_values highest entries
        - Normalize data
        """
        # compute gradient of the distance metric (with 1/_c gradient step size)
        grad_step = 1. / _c * _lambda * _left_side.T @ ((_lambda * _left_side @ S_old @ _right_side) - arr_X_target) @ _right_side.T

        # grad_step[np.abs(grad_step) < np.finfo(float).eps] = 0.
        # 1 step for minimizing + flatten necessary for the upcoming projection
        S_tmp = S_old - grad_step

        # projection # todo utiliser un paramètre à cette fonction? voir todo suivant
        # S_proj = projection_operator(S_tmp, _nb_keep_values)
        # inplace_hardthreshold(S_tmp, _nb_keep_values); S_proj = S_tmp
        S_proj = prox_splincol(S_tmp, _nb_keep_values / min(arr_X_target.shape)) # todo changer la façon dont les contraintes sont gérées (façon faµst c'est pas mal avec les lambda expressions)
        # normalize because all factors must have norm 1
        S_proj = S_proj / norm(S_proj, ord="fro")
        return S_proj

    def update_scaling_factor(X, X_est):
        return np.sum(X * X_est) / np.sum(X_est ** 2)

    def compute_objective_function(_f_lambda, _lst_S):
        return np.linalg.norm(arr_X_target - _f_lambda * multi_dot(_lst_S), ord='fro')

    logger.debug('Norme de arr_X_target: {}'.format(np.linalg.norm(arr_X_target, ord='fro')))
    assert len(lst_S_init) > 0
    assert get_side_prod(lst_S_init).shape == arr_X_target.shape
    assert len(lst_S_init) == nb_factors
    # initialization
    f_lambda = f_lambda_init
    lst_S = deepcopy(lst_S_init) # todo may not be necessary; check this ugliness

    objective_function = np.empty((nb_iter, nb_factors + 1))

    if update_right_to_left:
         # range arguments: start, stop, step
        factor_number_generator = range(-1, -(nb_factors + 1), -1)
    else:
        factor_number_generator = range(0, nb_factors, 1)
    # main loop
    for i_iter in range(nb_iter):
        # todo critère d'arrêt delta erreur = 10^-6
        for j in factor_number_generator:

            left_side = get_side_prod(lst_S[:j], arr_X_target.shape[0])  # L
            index_value_for_right_factors_selection = (nb_factors + j + 1) % (nb_factors + 1) # trust me, I am a scientist.
            right_side = get_side_prod(lst_S[index_value_for_right_factors_selection:], arr_X_target.shape[0])  # R

            # compute minimum c value (according to paper)
            min_c_value = (f_lambda * norm(right_side, ord=2) * norm(left_side, ord=2)) ** 2
            # add epsilon because it is exclusive minimum
            c = min_c_value * 1.001
            logger.debug("Lipsitchz constant value: {}; c value: {}".format(min_c_value, c))
            # compute new factor value
            lst_S[j] = update_S(lst_S[j], left_side, right_side, c, f_lambda,
                                 lst_nb_keep_values[j])

            if graphical_display:
                objective_function[i_iter, j - 1] = \
                    compute_objective_function(_f_lambda=f_lambda, _lst_S=lst_S)

        # re-compute the full factorisation
        if len(lst_S) == 1:
            arr_X_curr = lst_S[0]
        else:
            arr_X_curr = multi_dot(lst_S)
        # update lambda
        f_lambda = update_scaling_factor(arr_X_target, arr_X_curr)
        logger.debug("Lambda value: {}".format(f_lambda))

        objective_function[i_iter, -1] = \
            compute_objective_function(_f_lambda=f_lambda, _lst_S=lst_S)

    if graphical_display:
        plt.figure()
        plt.title("n factors {}".format(nb_factors))
        for j in range(nb_factors + 1):
            plt.semilogy(objective_function[:, j], label=str(j))
        plt.legend()
        plt.show()

        plt.figure()
        plt.semilogy(objective_function.flat)
        plt.legend()
        plt.show()

    return f_lambda, lst_S, arr_X_curr, objective_function


def HierarchicalPALM4MSA(arr_X_target: np.array,
                         lst_S_init: list,
                         nb_keep_values: int,
                         nb_iter: int,
                         f_lambda_init:float=1,
                         residual_sparsity_decrease=None,
                         residual_global_sparsity=None,
                         residual_on_right:bool=True,
                         update_right_to_left=True,
                         graphical_display=False):
    """


    :param arr_X_target:
    :param lst_S_init: The factors are given right to left. In all case.
    :param nb_keep_values:
    :param f_lambda_init:
    :param nb_iter:
    :param residual_sparsity_decrease:
    :param residual_global_sparsity:
    :param update_right_to_left: Way in which the factors are updated in the inner PALM4MSA algorithm. If update_right_to_left is True,
    the factors are updated right to left (e.g; the last factor in the list first). Otherwise the contrary.
    :param residual_on_right: During the split step, the residual can be computed as a right or left factor. If residual_on_right is True,
    the residuals are computed as right factors. We can also see this option as the update way for the hierarchical strategy:
    when the residual is computed on the right, it correspond to compute the last factor first (left to right according to the paper: the factor with the
    bigger number first)
    :return:
    """
    if update_right_to_left:
        raise NotImplementedError # todo voir pourquoi ça plante... zero division error?

    arr_residual = arr_X_target

    lst_S = deepcopy(lst_S_init)
    nb_factors = len(lst_S)

    if residual_sparsity_decrease is None:
        residual_sparsity_decrease = 0.5
    if residual_global_sparsity is None:
        residual_global_sparsity = min(arr_X_target.shape) ** 2

    nb_keep_values_relaxed = residual_global_sparsity
    f_lambda = f_lambda_init

    # main loop
    for k in range(nb_factors - 1):
        nb_factors_so_far = k + 1
        if nb_factors_so_far == nb_factors - 1:
            nb_keep_values_relaxed = nb_keep_values
        else:
            nb_keep_values_relaxed *= residual_sparsity_decrease
        logger.debug("Working on factor: {}".format(k))

        # calcule decomposition en 2 du résidu
        residual_init = get_side_prod(lst_S_init[nb_factors_so_far:])
        residual_init = np.zeros_like(residual_init)
        S_init = np.eye(lst_S_init[k].shape[0], lst_S_init[k].shape[1])

        logger.debug("Step split")

        func_split_step_palm4msa = lambda lst_S_init: PALM4MSA(
            arr_X_target=arr_residual,
            lst_S_init=lst_S_init, # eye for factor and zeros for residual
            nb_factors=2,
            lst_nb_keep_values=[nb_keep_values, int(nb_keep_values_relaxed)], #define constraints: ||0 = d pour T1; relaxed constraint on ||0 for T2
            f_lambda_init=1.,
            nb_iter=nb_iter,
            update_right_to_left=update_right_to_left)

        if update_right_to_left:
            lst_S_init_split_step = [S_init, residual_init]
        else:
            lst_S_init_split_step = [residual_init, S_init]

        if residual_on_right:
            f_lambda_prime, (new_factor, new_residual), _, _ =  func_split_step_palm4msa(lst_S_init=lst_S_init_split_step)
        else:
            f_lambda_prime, (new_factor, new_residual), _, _ = func_split_step_palm4msa(lst_S_init=lst_S_init_split_step)

        f_lambda *= f_lambda_prime
        if residual_on_right:
            lst_S[k] = new_factor
        else:
            lst_S[nb_factors-nb_factors_so_far] = new_factor

        if graphical_display:
            plt.figure()
            plt.subplot(221)
            plt.title('Input residual Iteration {}, etape split'.format(k))
            plt.imshow(arr_residual)
            plt.colorbar()

            plt.subplot(222)
            if residual_on_right:
                plt.imshow(f_lambda_prime * (new_factor @ new_residual))
                plt.title('lambda * new_factor @ new_residual')
            else:
                plt.imshow(f_lambda_prime * (new_residual @ new_factor))
                plt.title('lambda * new_residual @ new_factor')
            plt.colorbar()

            plt.subplot(223)
            plt.imshow(f_lambda_prime * new_factor)
            plt.colorbar()
            plt.title('lambda*new_factor')

            plt.subplot(224)
            plt.imshow(new_residual)
            plt.colorbar()
            plt.title('new_residual')

            plt.show()

        # get the k first elements [:k+1] and the next one (k+1)th as arr_residual (depend on the residual_on_right option)
        logger.debug("Step finetuning")

        func_fine_tune_step_palm4msa = lambda lst_S_init, lst_nb_keep_values: PALM4MSA(
            arr_X_target=arr_X_target,
            lst_S_init=lst_S_init,  # lst_S[:nb_factors_so_far] + [new_residual],
            nb_factors=nb_factors_so_far + 1,
            lst_nb_keep_values=lst_nb_keep_values, # lst_nb_keep_values_constraints,
            f_lambda_init=f_lambda,
            nb_iter=nb_iter,
            update_right_to_left=update_right_to_left)

        if residual_on_right:
            f_lambda, (*lst_S[:nb_factors_so_far], arr_residual), _, _ = func_fine_tune_step_palm4msa(lst_S_init=lst_S[:nb_factors_so_far] + [new_residual],
                                                                                                      lst_nb_keep_values=[nb_keep_values] * nb_factors_so_far + [int(nb_keep_values_relaxed)])
        else:
            f_lambda, (arr_residual, *lst_S[-nb_factors_so_far:]), _, _ = func_fine_tune_step_palm4msa(lst_S_init=[new_residual] + lst_S[-nb_factors_so_far:],
                                                                                                      lst_nb_keep_values=[int(nb_keep_values_relaxed)] + [nb_keep_values] * nb_factors_so_far)

        if graphical_display:
            plt.figure()
            plt.subplot(221)
            plt.title('Residual Iteration {}, step fine tune '.format(k))
            plt.imshow(arr_residual)
            plt.colorbar()

            plt.subplot(222)
            plt.imshow(f_lambda * get_side_prod(
                lst_S[:nb_factors_so_far] + [arr_residual]))
            plt.colorbar()
            plt.title('reconstructed')


            plt.subplot(223)
            plt.imshow(lst_S[k])
            plt.colorbar()
            plt.title('current factor')


            plt.subplot(224)
            plt.imshow(arr_residual)
            plt.colorbar()
            plt.title('residual (right factor)')

            plt.show()

    # last factor is residual of last palm4LED
    if residual_on_right:
        lst_S[-1] = arr_residual
    else:
        lst_S[0] = arr_residual


    if len(lst_S) == 1:
        arr_X_curr = f_lambda * lst_S[0]
    else:
        arr_X_curr = f_lambda * multi_dot(lst_S)

    return f_lambda, lst_S, arr_X_curr
