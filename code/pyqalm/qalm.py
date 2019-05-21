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
from scipy.sparse import coo_matrix

from pyqalm.projection_operators import prox_splincol, inplace_hardthreshold
from pyqalm.utils import get_side_prod, logger
from pyqalm.data_structures import SparseFactors


def compute_objective_function(arr_X_target, _f_lambda, _lst_S):
    if isinstance(_lst_S, SparseFactors):
        reconstruct = _f_lambda * _lst_S.compute_product()
    else:
        reconstruct = _f_lambda * multi_dot(_lst_S)
    return np.linalg.norm(arr_X_target - reconstruct, ord='fro') ** 2


def PALM4MSA(arr_X_target: np.array,
             lst_S_init: list,
             nb_factors: int,
             lst_projection_functions: list,
             f_lambda_init: float,
             nb_iter: int,
             update_right_to_left=True,
             graphical_display=False):
    """
    lst S init contains factors in decreasing indexes (e.g: the order along which they are multiplied in the product).
        example: S5 S4 S3 S2 S1

    lst S [-j] = Sj

    """

    def update_S(S_old, _left_side, _right_side, _c, _lambda,
                 projection_function):
        """
        Return the new factor value.

        - Compute gradient
        - Do gradient step
        - Project data on _nb_keep_values highest entries
        - Normalize data
        """
        # compute gradient of the distance metric (with 1/_c gradient step size)
        grad_step = 1. / _c * _lambda \
                    * _left_side.T \
                    @ ((_lambda * _left_side @ S_old @ _right_side)
                       - arr_X_target) \
                    @ _right_side.T

        # grad_step[np.abs(grad_step) < np.finfo(float).eps] = 0.
        # 1 step for minimizing + flatten necessary for the upcoming projection
        S_tmp = S_old - grad_step

        # normalize because all factors must have norm 1
        S_proj = projection_function(S_tmp)
        S_proj = S_proj / norm(S_proj, ord="fro")
        return S_proj

    def update_scaling_factor(X, X_est):
        return np.sum(X * X_est) / np.sum(X_est ** 2)

    logger.debug('Norme de arr_X_target: {}'.format(
        np.linalg.norm(arr_X_target, ord='fro')))
    assert len(lst_S_init) > 0
    assert get_side_prod(lst_S_init).shape == arr_X_target.shape
    assert len(lst_S_init) == nb_factors
    # initialization
    f_lambda = f_lambda_init
    lst_S = deepcopy(
        lst_S_init)  # todo may not be necessary; check this ugliness

    objective_function = np.empty((nb_iter, nb_factors + 1))

    if update_right_to_left:
        # range arguments: start, stop, step
        factor_number_generator = range(-1, -(nb_factors + 1), -1)
    else:
        factor_number_generator = range(0, nb_factors, 1)
    # main loop
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while i_iter == 0 or ((i_iter < nb_iter) and (
            delta_objective_error > delta_objective_error_threshold)):

        for j in factor_number_generator:
            if lst_projection_functions[j].__name__ == "constant_proj":
                continue

            left_side = get_side_prod(lst_S[:j], (
                arr_X_target.shape[0], arr_X_target.shape[0]))  # L
            index_value_for_right_factors_selection = (nb_factors + j + 1) % (
                    nb_factors + 1)  # trust me, I am a scientist.
            right_side = get_side_prod(
                lst_S[index_value_for_right_factors_selection:],
                (arr_X_target.shape[1], arr_X_target.shape[1]))  # R

            # compute minimum c value (according to paper)
            min_c_value = (f_lambda * norm(right_side, ord=2)
                           * norm(left_side, ord=2)) ** 2
            # add epsilon because it is exclusive minimum
            c = min_c_value * 1.001
            logger.debug(
                "Lipsitchz constant value: {}; c value: {}".format(min_c_value,
                                                                   c))
            # compute new factor value
            lst_S[j] = update_S(lst_S[j], left_side, right_side, c, f_lambda,
                                lst_projection_functions[j])

            if graphical_display:
                objective_function[i_iter, j - 1] = \
                    compute_objective_function(arr_X_target,
                                               _f_lambda=f_lambda,
                                               _lst_S=lst_S)

        # re-compute the full factorisation
        if len(lst_S) == 1:
            arr_X_curr = lst_S[0]
        else:
            arr_X_curr = multi_dot(lst_S)
        # update lambda
        f_lambda = update_scaling_factor(arr_X_target, arr_X_curr)
        logger.debug("Lambda value: {}".format(f_lambda))

        objective_function[i_iter, -1] = \
            compute_objective_function(arr_X_target, _f_lambda=f_lambda,
                                       _lst_S=lst_S)

        logger.debug("Iteration {}; Objective value: {}"
                     .format(i_iter, objective_function[i_iter, -1]))

        if i_iter >= 1:
            delta_objective_error = np.abs(
                objective_function[i_iter, -1] - objective_function[
                    i_iter - 1, -1]) / objective_function[
                                        i_iter - 1, -1]
            # TODO vérifier que l'erreur absolue est plus petite que le
            # threshold plusieurs fois d'affilée

        i_iter += 1

    objective_function = objective_function[:i_iter, :]

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

    # todo maybe change arrX_curr by lambda * arrX_curr
    return f_lambda, lst_S, arr_X_curr, objective_function, i_iter


def HierarchicalPALM4MSA(arr_X_target: np.array,
                         lst_S_init: list,
                         lst_dct_projection_function: list,
                         nb_iter: int,
                         f_lambda_init: float = 1,
                         residual_on_right: bool = True,
                         update_right_to_left=True,
                         graphical_display=False):
    """


    :param arr_X_target:
    :param lst_S_init: The factors are given right to left. In all case.
    :param nb_keep_values:
    :param f_lambda_init:
    :param nb_iter:
    :param update_right_to_left: Way in which the factors are updated in the inner PALM4MSA algorithm. If update_right_to_left is True,
    the factors are updated right to left (e.g; the last factor in the list first). Otherwise the contrary.
    :param residual_on_right: During the split step, the residual can be computed as a right or left factor. If residual_on_right is True,
    the residuals are computed as right factors. We can also see this option as the update way for the hierarchical strategy:
    when the residual is computed on the right, it correspond to compute the last factor first (left to right according to the paper: the factor with the
    bigger number first)
    :return:
    """
    if not update_right_to_left:
        raise NotImplementedError  # todo voir pourquoi ça plante... mismatch dimension

    # min_shape = min(arr_X_target.shape)

    arr_residual = arr_X_target

    lst_S = deepcopy(lst_S_init)
    nb_factors = len(lst_S)

    # check if lst_dct_param_projection_operator contains a list of dict with param for step split and finetune
    assert len(lst_dct_projection_function) == nb_factors - 1
    assert all(
        len({"split", "finetune"}.difference(dct.keys())) == 0 for dct in
        lst_dct_projection_function)

    lst_nb_iter_by_factor = []

    f_lambda = f_lambda_init # todo enlever?

    objective_function = np.empty((nb_factors, 3))

    # main loop
    for k in range(nb_factors - 1):
        nb_factors_so_far = k + 1

        logger.info("Working on factor: {}".format(k))
        logger.info("Step split")

        objective_function[k, 0] = compute_objective_function(arr_X_target,
                                                              f_lambda, lst_S)

        # calcule decomposition en 2 du résidu précédent
        if k == 0:
            f_lambda_init_split = f_lambda_init
        else:
            f_lambda_init_split = 1.

        func_split_step_palm4msa = lambda lst_S_init: PALM4MSA(
            arr_X_target=arr_residual,
            lst_S_init=lst_S_init,  # eye for factor and zeros for residual
            nb_factors=2,
            lst_projection_functions=lst_dct_projection_function[k]["split"], #define constraints: ||0 = d pour T1; relaxed constraint on ||0 for T2
            f_lambda_init=f_lambda_init_split,
            nb_iter=nb_iter,
            update_right_to_left=update_right_to_left,
            graphical_display=graphical_display)

        if residual_on_right:
            residual_init = get_side_prod(lst_S_init[nb_factors_so_far:])
            S_init = lst_S_init[k]
            lst_S_init_split_step = [S_init, residual_init]

        else:
            residual_init = get_side_prod(lst_S_init[:-nb_factors_so_far])
            S_init = lst_S_init[-nb_factors_so_far]
            lst_S_init_split_step = [residual_init, S_init]

        if residual_on_right:
            f_lambda_prime, (new_factor, new_residual), unscaled_residual_reconstruction, _, nb_iter_this_factor = func_split_step_palm4msa(lst_S_init=lst_S_init_split_step)
        else:
            f_lambda_prime, (new_residual, new_factor), unscaled_residual_reconstruction, _, nb_iter_this_factor = func_split_step_palm4msa(lst_S_init=lst_S_init_split_step)

        if k == 0:
            f_lambda = f_lambda_prime
            # f_lambda = f_lambda
        else:
            f_lambda *= f_lambda_prime

        if residual_on_right:
            lst_S[k] = new_factor
        else:
            lst_S[nb_factors - nb_factors_so_far] = new_factor

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
        logger.info("Step finetuning")

        objective_function[k, 1] = compute_objective_function(arr_X_target,
                                                              f_lambda, lst_S)


        func_fine_tune_step_palm4msa = lambda lst_S_init: PALM4MSA(
            arr_X_target=arr_X_target,
            lst_S_init=lst_S_init,
            nb_factors=nb_factors_so_far + 1,
            lst_projection_functions=lst_dct_projection_function[k][
                "finetune"],
            f_lambda_init=f_lambda,
            nb_iter=nb_iter,
            update_right_to_left=update_right_to_left,
            graphical_display=graphical_display)

        if residual_on_right:
            f_lambda, (*lst_S[:nb_factors_so_far],
                       arr_residual), _, _, nb_iter_this_factor_bis = func_fine_tune_step_palm4msa(
                lst_S_init=lst_S[:nb_factors_so_far] + [new_residual])
        else:
            f_lambda, (arr_residual, *lst_S[
                                      -nb_factors_so_far:]), _, _, nb_iter_this_factor_bis = func_fine_tune_step_palm4msa(
                lst_S_init=[new_residual] + lst_S[-nb_factors_so_far:])

        lst_nb_iter_by_factor.append(
            nb_iter_this_factor + nb_iter_this_factor_bis)

        objective_function[k, 2] = compute_objective_function(arr_X_target,
                                                              f_lambda, lst_S)

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

    objective_function[nb_factors - 1, :] = np.array(
        [compute_objective_function(arr_X_target, f_lambda, lst_S)] * 3)

    if len(lst_S) == 1:
        arr_X_curr = f_lambda * lst_S[0]
    else:
        arr_X_curr = f_lambda * multi_dot(lst_S)

    return f_lambda, lst_S, arr_X_curr, lst_nb_iter_by_factor, objective_function


def palm4msa_fast1(arr_X_target: np.array,
                   lst_S_init: list,
                   nb_factors: int,
                   lst_projection_functions: list,
                   f_lambda_init: float,
                   nb_iter: int,
                   update_right_to_left=True,
                   graphical_display=False):
    """
    lst S init contains factors in decreasing indexes (e.g: the order along which they are multiplied in the product).
        example: S5 S4 S3 S2 S1

    lst S [-j] = Sj

    """

    def update_S(S_old, _L, _R, _c, _lambda, projection_function):
        """
        Return the new factor value.

        - Compute gradient
        - Do gradient step
        - Project data on _nb_keep_values highest entries
        - Normalize data
        """
        # compute gradient of the distance metric (with 1/_c gradient step size)
        S = SparseFactors(_L.get_list_of_factors() + [S_old]
                          + _R.get_list_of_factors())
        res = _lambda * S.compute_product() - arr_X_target
        res_RT = _R.dot(res.T).T if _R.n_factors > 0 else res
        LT_res_RT = _L.transpose().dot(res_RT) if _L.n_factors > 0 else res_RT
        grad_step = 1. / _c * _lambda * LT_res_RT

        # grad_step[np.abs(grad_step) < np.finfo(float).eps] = 0.
        # 1 step for minimizing + flatten necessary for the upcoming projection
        S_tmp = S_old - grad_step

        # normalize because all factors must have norm 1
        S_proj = projection_function(S_tmp)
        S_proj = S_proj / norm(S_proj, ord="fro")
        return S_proj

    def update_scaling_factor(X, X_est):
        return np.sum(X * X_est) / np.sum(X_est ** 2)

    logger.debug('Norme de arr_X_target: {}'.format(
        np.linalg.norm(arr_X_target, ord='fro')))
    assert len(lst_S_init) > 0
    assert get_side_prod(lst_S_init).shape == arr_X_target.shape
    assert len(lst_S_init) == nb_factors
    # initialization
    f_lambda = f_lambda_init
    S_factors_op = SparseFactors(lst_S_init)

    objective_function = np.empty((nb_iter, nb_factors + 1))

    if update_right_to_left:
        # range arguments: start, stop, step
        factor_number_generator = range(-1, -(nb_factors + 1), -1)
    else:
        factor_number_generator = range(0, nb_factors, 1)
    # main loop
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while i_iter == 0 or ((i_iter < nb_iter) and (
            delta_objective_error > delta_objective_error_threshold)):

        for j in factor_number_generator:
            if lst_projection_functions[j].__name__ == "constant_proj":
                continue

            index_value_for_right_factors_selection = (nb_factors + j + 1) % (
                    nb_factors + 1)  # trust me, I am a scientist.
            lst_factors = S_factors_op.get_list_of_factors()
            L = SparseFactors(lst_factors[:j])
            R = SparseFactors(
                lst_factors[index_value_for_right_factors_selection:])

            # compute minimum c value (according to paper)
            L_norm = L.compute_spectral_norm() if L.n_factors > 0 else 1
            R_norm = R.compute_spectral_norm() if R.n_factors > 0 else 1
            min_c_value = (f_lambda * L_norm * R_norm) ** 2
            # add epsilon because it is exclusive minimum
            c = min_c_value * 1.001
            logger.debug(
                "Lipsitchz constant value: {}; c value: {}".format(min_c_value,
                                                                   c))
            # compute new factor value
            Sj = update_S(S_factors_op.get_factor(j), L, R, c, f_lambda,
                          lst_projection_functions[j])
            S_factors_op.set_factor(j, Sj)

            if graphical_display:
                objective_function[i_iter, j - 1] = \
                    compute_objective_function(arr_X_target,
                                               _f_lambda=f_lambda,
                                               _lst_S=S_factors_op)

        # re-compute the full factorisation
        if S_factors_op.n_factors == 1:
            arr_X_curr = S_factors_op.get_factor(0)
        else:
            arr_X_curr = S_factors_op.compute_product()
        # update lambda
        f_lambda = update_scaling_factor(arr_X_target, arr_X_curr)
        logger.debug("Lambda value: {}".format(f_lambda))

        objective_function[i_iter, -1] = \
            compute_objective_function(arr_X_target, _f_lambda=f_lambda,
                                       _lst_S=S_factors_op)

        logger.debug("Iteration {}; Objective value: {}"
                     .format(i_iter, objective_function[i_iter, -1]))

        if i_iter >= 1:
            delta_objective_error = np.abs(
                objective_function[i_iter, -1] - objective_function[
                    i_iter - 1, -1]) / objective_function[i_iter - 1, -1]
            # TODO vérifier que l'erreur absolue est plus petite que le
            # threshold plusieurs fois d'affilée

        i_iter += 1

    objective_function = objective_function[:i_iter, :]

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

    return f_lambda, S_factors_op, arr_X_curr, objective_function, i_iter


def palm4msa_fast2(arr_X_target: np.array,
                   lst_S_init: list,
                   nb_factors: int,
                   lst_projection_functions: list,
                   f_lambda_init: float,
                   nb_iter: int,
                   update_right_to_left=True,
                   graphical_display=False):
    """
    lst S init contains factors in decreasing indexes (e.g: the order along which they are multiplied in the product).
        example: S5 S4 S3 S2 S1

    lst S [-j] = Sj

    """
    logger.debug('Norme de arr_X_target: {}'.format(
        np.linalg.norm(arr_X_target, ord='fro')))
    assert len(lst_S_init) > 0
    assert get_side_prod(lst_S_init).shape == arr_X_target.shape
    assert len(lst_S_init) == nb_factors
    # initialization
    f_lambda = f_lambda_init
    S_factors_op = SparseFactors(lst_S_init)

    objective_function = np.empty((nb_iter, nb_factors + 1))

    if update_right_to_left:
        # range arguments: start, stop, step
        factor_number_generator = range(-1, -(nb_factors + 1), -1)
    else:
        factor_number_generator = range(0, nb_factors, 1)
    # main loop
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    while i_iter == 0 or ((i_iter < nb_iter) and (
            delta_objective_error > delta_objective_error_threshold)):

        for j in factor_number_generator:
            if lst_projection_functions[j].__name__ == "constant_proj":
                continue

            index_value_for_right_factors_selection = (nb_factors + j + 1) % (
                    nb_factors + 1)  # trust me, I am a scientist.
            lst_factors = S_factors_op.get_list_of_factors()
            L = SparseFactors(lst_factors[:j])
            R = SparseFactors(
                lst_factors[index_value_for_right_factors_selection:])

            # compute minimum c value (according to paper)
            L_norm = L.compute_spectral_norm() if L.n_factors > 0 else 1
            R_norm = R.compute_spectral_norm() if R.n_factors > 0 else 1
            min_c_value = (f_lambda * L_norm * R_norm) ** 2
            # add epsilon because it is exclusive minimum
            c = min_c_value * 1.001
            logger.debug(
                "Lipsitchz constant value: {}; c value: {}".format(min_c_value,
                                                                   c))
            # compute new factor value
            res = f_lambda * S_factors_op.compute_product() - arr_X_target
            res_RT = R.dot(res.T).T if R.n_factors > 0 else res
            LT_res_RT = L.transpose().dot(res_RT) if L.n_factors > 0 else res_RT
            grad_step = 1. / c * f_lambda * LT_res_RT

            Sj = S_factors_op.get_factor(j)

            # normalize because all factors must have norm 1
            S_proj = coo_matrix(lst_projection_functions[j](Sj - grad_step))
            S_proj /= np.sqrt(S_proj.power(2).sum())

            S_factors_op.set_factor(j, S_proj)

            if graphical_display:
                objective_function[i_iter, j - 1] = \
                    compute_objective_function(arr_X_target,
                                               _f_lambda=f_lambda,
                                               _lst_S=S_factors_op)

        # re-compute the full factorisation
        if S_factors_op.n_factors == 1:
            arr_X_curr = S_factors_op.get_factor(0)
        else:
            arr_X_curr = S_factors_op.compute_product()
        # update lambda
        f_lambda = np.sum(arr_X_target * arr_X_curr) / np.sum(arr_X_curr ** 2)

        logger.debug("Lambda value: {}".format(f_lambda))

        objective_function[i_iter, -1] = \
            compute_objective_function(arr_X_target, _f_lambda=f_lambda,
                                       _lst_S=S_factors_op)

        logger.debug("Iteration {}; Objective value: {}"
                     .format(i_iter, objective_function[i_iter, -1]))

        if i_iter >= 1:
            delta_objective_error = np.abs(
                objective_function[i_iter, -1] - objective_function[
                    i_iter - 1, -1]) / objective_function[i_iter - 1, -1]
            # TODO vérifier que l'erreur absolue est plus petite que le
            # threshold plusieurs fois d'affilée

        i_iter += 1

    objective_function = objective_function[:i_iter, :]

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

    return f_lambda, S_factors_op, arr_X_curr, objective_function, i_iter


if __name__ == '__main__':
    from scipy.linalg import hadamard
    from pyqalm.utils import get_lambda_proxsplincol

    data = dict()
    data['hadamard'] = hadamard(32)

    n_rows = 64
    n_cols = 77
    X = np.random.randn(n_rows, n_cols)
    data['random matrix'] = X

    for k, X in data.items():

        d = np.min(X.shape)
        if X.shape[1] == d:
            X = X.T
        nb_factors = int(np.log2(d))
        lst_S_init = []
        for _ in range(nb_factors - 1):
            lst_S_init.append(np.eye(d))
        lst_S_init.append(np.zeros(X.shape))

        nb_keep_values = 2 * d
        nb_values_residual = int(d / 2 ** nb_factors) * d
        lst_projection_functions = \
            [get_lambda_proxsplincol(nb_keep_values)] * nb_factors \
            + [get_lambda_proxsplincol(nb_values_residual)]

        f_lambda_init = 1
        nb_iter = 10
        update_right_to_left = True
        graphical_display = False
        f_lambda_ref, lst_S_ref, arr_X_curr_ref, objective_function_ref, \
        i_iter_ref = \
            PALM4MSA(X,
                     lst_S_init=lst_S_init,
                     nb_factors=nb_factors,
                     lst_projection_functions=lst_projection_functions,
                     f_lambda_init=f_lambda_init,
                     nb_iter=nb_iter,
                     update_right_to_left=update_right_to_left,
                     graphical_display=graphical_display)

        f_lambda, lst_S, arr_X_curr, objective_function, i_iter = \
            palm4msa_fast1(X,
                           lst_S_init=lst_S_init,
                           nb_factors=nb_factors,
                           lst_projection_functions=lst_projection_functions,
                           # lst_projection_functions=lst_projection_functions_fast,
                           f_lambda_init=f_lambda_init,
                           nb_iter=nb_iter,
                           update_right_to_left=update_right_to_left,
                           graphical_display=graphical_display)

