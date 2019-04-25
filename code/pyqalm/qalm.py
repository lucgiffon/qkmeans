# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon
"""
import numpy as np
from numpy.linalg import norm
from numpy import identity
from numpy import argpartition
from numpy.linalg import multi_dot
from copy import deepcopy

import matplotlib.pyplot as plt  # TODO remove (no graphics in algorithms)


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


def update_scaling_factor(X, X_est):
    return np.sum(X * X_est) / np.sum(X_est**2)


def PALM4MSA(arr_X_target: np.array,
             lst_S_init: list,
             nb_factors: int,
             lst_nb_keep_values: list,
             f_lambda_init: float,
             nb_iter: int):
    """
    lst S init contains factors in decreasing indexes: S5 S4 S3 S2 S1

    lst S [-j] = Sj

    """
    print('Norme de arr_X_target:', np.linalg.norm(arr_X_target, ord='fro'))

    def update_S(S_old, _left_side, _right_side, _c, _lambda, _nb_keep_values):
        """
        Return the new factor value.

        - Compute gradient
        - Do gradient step
        - Project data on _nb_keep_values highest entries
        - Normalize data
        """
        # compute gradient of the distance metric (with 1/_c gradient step size)
        grad_step = 1. / _c * _lambda * _left_side.T @ ((
                                                                    _lambda * _left_side @ S_old @ _right_side) - arr_X_target) @ _right_side.T

        # grad_step[np.abs(grad_step) < np.finfo(float).eps] = 0.
        # 1 step for minimizing + flatten necessary for the upcoming projection
        S_tmp = S_old - grad_step
        # projection
        # S_proj = projection_operator(S_tmp, _nb_keep_values)
        inplace_hardthreshold(S_tmp, _nb_keep_values)
        # normalize because all factors must have norm 1
        S_proj = S_tmp
        S_proj = S_proj / norm(S_proj, ord="fro")
        return S_proj

    assert len(lst_S_init) > 0
    assert get_side_prod(lst_S_init).shape == arr_X_target.shape
    assert len(lst_S_init) == nb_factors
    # initialization
    f_lambda = f_lambda_init
    lst_S = deepcopy(lst_S_init)
    #     arr_X_curr = multi_dot(lst_S) # modified by VE
    #     f_lambda = np.linalg.norm(arr_X_target, ord='fro') / np.linalg.norm(arr_X_curr, ord='fro') # modified by VE
    #     f_lambda = trace(arr_X_target.T @ arr_X_curr) / trace(arr_X_curr.T @ arr_X_curr) # modified by VE

    objective_function = np.empty((nb_iter, nb_factors + 1))
    # main loop
    for i_iter in range(nb_iter):
        #         print('Norme de la factorisation:', np.linalg.norm(f_lambda * multi_dot(lst_S), ord='fro'))
        # todo critère d'arrêt delta erreur = 10^-6
        for j in range(1, nb_factors + 1):
            # left_side = get_side_prod(lst_S[:j], arr_X_target.shape[0]) # L = products of all yet updated factors during this iter
            # right_side = get_side_prod(lst_S[j+1:], arr_X_target.shape[0]) # R = products of all not yet updated factors

            left_side = get_side_prod(lst_S[:-j], arr_X_target.shape[0])  # L
            right_side = get_side_prod(lst_S[nb_factors - j + 1:],
                                       arr_X_target.shape[0])  # R
            #             print('j: {}/{}'.format(j, nb_factors))
            #             print('Left side', lst_S[:-j], len(lst_S[:-j]))
            #             print(norm(left_side, ord=2))
            #             print(lst_S[-j])
            #             print('Right side', lst_S[nb_factors-j+1:], len(lst_S[nb_factors-j+1:]))
            #             print(norm(right_side, ord=2))

            # compute minimum c value (according to paper)
            min_c_value = (f_lambda * norm(right_side, ord=2) * norm(left_side,
                                                                     ord=2)) ** 2
            # add epsilon because it is exclusive minimum
            # c = min_c_value * (1+np.finfo(float).eps)
            c = min_c_value * 1.001
            # c = min_c_value + 1

            # compute new factor value
            lst_S[-j] = update_S(lst_S[-j], left_side, right_side, c, f_lambda,
                                 lst_nb_keep_values[-j])

            objective_function[i_iter, j - 1] = np.linalg.norm(
                arr_X_target - f_lambda * multi_dot(lst_S), ord='fro')

        # re-compute the full factorisation
        if len(lst_S) == 1:
            arr_X_curr = lst_S[0]
        else:
            arr_X_curr = multi_dot(lst_S)
        # update lambda
        f_lambda = update_scaling_factor(arr_X_target, arr_X_curr)

        objective_function[i_iter, -1] = np.linalg.norm(
            arr_X_target - f_lambda * multi_dot(lst_S), ord='fro')

    plt.figure()
    for j in range(nb_factors + 1):
        plt.semilogy(objective_function[:, j], label=str(j))
    plt.legend()
    plt.show()
    return f_lambda, lst_S, arr_X_curr, objective_function


def HierarchicalPALM4MSA(arr_X_target: np.array,
                         lst_S_init: list,
                         nb_keep_values: int,
                         f_lambda_init: float,
                         nb_iter: int,

                         residual_sparsity_decrease=None,
                         residual_global_sparsity=None,
                         right_to_left=True):
    # initialisation
    if right_to_left:
        arr_residual = arr_X_target
    else:  # attention: vérifier l'équivalence en prenant en compte le lambda
        arr_residual = arr_X_target.T
        lst_S_init = [S.T for S in lst_S_init[::-1]]

    lst_S = deepcopy(lst_S_init)
    nb_factors = len(lst_S)

    if residual_sparsity_decrease is None:
        residual_sparsity_decrease = 0.5
    if residual_global_sparsity is None:
        residual_global_sparsity = min(arr_X_target.shape) ** 2

    nb_keep_values_relaxed = residual_global_sparsity

    # main loop
    for k in range(nb_factors - 1):
        nb_factors_so_far = k + 1
        if nb_factors_so_far == nb_factors - 1:
            nb_keep_values_relaxed = nb_keep_values
        else:
            nb_keep_values_relaxed *= residual_sparsity_decrease
        print("working on factor:", k)
        # define constraints: ||0 = d pour T1; no constraint on ||0 for T2
        lst_nb_keep_values_constraints = [int(nb_keep_values_relaxed),
                                          nb_keep_values]
        # calcule decomposition en 2 du dernier résidu de l'opération précédente
        residual_init = get_side_prod(lst_S_init[:-nb_factors_so_far])
        f_lambda_prime, (F2, F1), _, _ = PALM4MSA(arr_X_target=arr_residual,
                                                  lst_S_init=[residual_init,
                                                              lst_S_init[
                                                                  -nb_factors_so_far]],
                                                  nb_factors=2,
                                                  lst_nb_keep_values=lst_nb_keep_values_constraints,
                                                  f_lambda_init=f_lambda_init,
                                                  nb_iter=nb_iter)

        lst_S[-nb_factors_so_far] = F1

        print("1er appel")
        print("residu:")
        plt.imshow(f_lambda_prime * F2)
        plt.show()
        print("F1:")
        plt.imshow(F1)
        plt.show()
        print("F2F1:")
        plt.imshow(f_lambda_prime * (F2 @ F1))
        plt.show()

        # arr_residual = F2
        # get the k first elements [:k+1] and the next one (k+1)th as arr_residual
        lst_nb_keep_values_constraints = [int(nb_keep_values_relaxed)] + [
            nb_keep_values] * nb_factors_so_far
        f_lambda, (arr_residual, *lst_S[-nb_factors_so_far:]), _, _ = PALM4MSA(
            arr_X_target=arr_X_target,
            lst_S_init=[F2] + lst_S[-nb_factors_so_far:],
            nb_factors=nb_factors_so_far + 1,
            lst_nb_keep_values=lst_nb_keep_values_constraints,
            f_lambda_init=f_lambda_prime,
            nb_iter=nb_iter)
        print("2eme appel")
        print("residu:")
        plt.imshow(arr_residual)
        plt.show()
        print("current factor:")
        plt.imshow(lst_S[-nb_factors_so_far])
        plt.show()
        print("reconstructed:")
        plt.imshow(f_lambda_prime * get_side_prod(
            [arr_residual] + lst_S[-nb_factors_so_far:]))
        plt.show()
        # arr_residual = lst_S[k+1]
        # arr_residual = T2

    #         print(f_lambda_prime)
    # last factor is residual of last palm4LED
    lst_S[0] = arr_residual
    if not right_to_left:
        lst_S = [S.T for S in lst_S[::-1]]

    if len(lst_S) == 1:
        arr_X_curr = f_lambda * lst_S[0]
    else:
        arr_X_curr = f_lambda * multi_dot(lst_S)

    return f_lambda, lst_S, arr_X_curr