# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon
"""
from copy import deepcopy

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from pyqalm.palm.utils import compute_objective_function, update_scaling_factor
from scipy.sparse import coo_matrix

from pyqalm.utils import get_side_prod, logger
from pyqalm.data_structures import SparseFactors
from sklearn import datasets


def hierarchical_palm4msa(arr_X_target: np.array,
                          lst_S_init: list,
                          lst_dct_projection_function: list,
                          nb_iter: int,
                          f_lambda_init: float = 1,
                          residual_on_right: bool = True,
                          update_right_to_left=True,
                          track_objective_palm=False,
                          return_objective_function=False,
                          delta_objective_error_threshold_palm=1e-6):
    """


    :param arr_X_target:
    :param lst_S_init: The factors are given right to left. In all case.
    :param nb_keep_values:
    :param f_lambda_init:
    :param nb_iter:
    :param update_right_to_left: Way in which the factors are updated in the inner palm4msa algorithm. If update_right_to_left is True,
    the factors are updated right to left (e.g; the last factor in the list first). Otherwise the contrary.
    :param residual_on_right: During the split step, the residual can be computed as a right or left factor. If residual_on_right is True,
    the residuals are computed as right factors. We can also see this option as the update way for the hierarchical strategy:
    when the residual is computed on the right, it correspond to compute the last factor first (left to right according to the paper: the factor with the
    bigger number first)
    :return:
    """
    if not update_right_to_left:
        raise NotImplementedError  # todo voir pourquoi ça plante... mismatch dimension

    arr_residual = arr_X_target

    op_S_factors = SparseFactors(deepcopy(lst_S_init))
    nb_factors = op_S_factors.n_factors

    # check if lst_dct_param_projection_operator contains a list of dict with param for step split and finetune
    assert len(lst_dct_projection_function) == nb_factors - 1
    assert all(
        len({"split", "finetune"}.difference(dct.keys())) == 0 for dct in
        lst_dct_projection_function)

    f_lambda = f_lambda_init

    if return_objective_function:
        objective_function = np.empty((nb_factors, 3))
    else:
        objective_function = None

    lst_objectives = []

    # main loop
    for k in range(nb_factors - 1):
        lst_objective_split_fine_fac_k = []

        nb_factors_so_far = k + 1

        logger.info("Working on factor: {}".format(k))
        logger.info("Step split")

        ########################## Step split ##########################################################

        if return_objective_function:
            # compute objective before split step
            objective_function[k, 0] = compute_objective_function(arr_X_target, f_lambda, op_S_factors)

        # calcule decomposition en 2 du résidu précédent
        if k == 0:
            f_lambda_init_split = f_lambda_init
        else:
            f_lambda_init_split = 1.

        func_split_step_palm4msa = lambda lst_S_init: palm4msa(
            arr_X_target=arr_residual,
            lst_S_init=lst_S_init,  # eye for factor and zeros for residual
            nb_factors=2,
            lst_projection_functions=lst_dct_projection_function[k]["split"],
            # define constraints: ||0 = d pour T1; relaxed constraint on ||0 for T2
            f_lambda_init=f_lambda_init_split,
            nb_iter=nb_iter,
            update_right_to_left=update_right_to_left,
            track_objective=track_objective_palm,
            delta_objective_error_threshold=delta_objective_error_threshold_palm
            )

        if residual_on_right:
            op_S_factors_init = SparseFactors(lst_S_init[nb_factors_so_far:])
            residual_init = op_S_factors_init.compute_product() # todo I think this product can be prepared before and save computation
            lst_S_init_split_step = [lst_S_init[k], residual_init]
            f_lambda_prime, S_out, unscaled_residual_reconstruction, objective_palm_split, _ = \
                func_split_step_palm4msa(lst_S_init=lst_S_init_split_step)
            new_factor = S_out.get_factor(0)
            new_residual = S_out.get_factor(1)
            op_S_factors.set_factor(k, new_factor)

        else:
            op_S_factors_init = SparseFactors(lst_S_init[:-nb_factors_so_far])
            residual_init = op_S_factors_init.compute_product() # todo I think this product can be prepared before and save computation
            lst_S_init_split_step = [residual_init,
                                     lst_S_init[-nb_factors_so_far]]
            f_lambda_prime, S_out, unscaled_residual_reconstruction, objective_palm_split, _ = \
                func_split_step_palm4msa(lst_S_init=lst_S_init_split_step)
            new_residual = S_out.get_factor(0)
            new_factor = S_out.get_factor(1)
            op_S_factors.set_factor(nb_factors - nb_factors_so_far, new_factor)


        if k == 0:
            f_lambda = f_lambda_prime
        else:
            f_lambda *= f_lambda_prime

        lst_objective_split_fine_fac_k.append(objective_palm_split)

        # get the k first elements [:k+1] and the next one (k+1)th as arr_residual (depend on the residual_on_right option)
        logger.info("Step finetuning")

        ########################## Step finetuning ##########################################################

        if return_objective_function:
            objective_function[k, 1] = compute_objective_function(arr_X_target,
                                                                  f_lambda,
                                                                  op_S_factors)

        func_fine_tune_step_palm4msa = lambda lst_S_init: palm4msa(
            arr_X_target=arr_X_target,
            lst_S_init=lst_S_init,
            nb_factors=nb_factors_so_far + 1,
            lst_projection_functions=lst_dct_projection_function[k][
                "finetune"],
            f_lambda_init=f_lambda,
            nb_iter=nb_iter,
            update_right_to_left=update_right_to_left,
            track_objective=track_objective_palm,
            delta_objective_error_threshold=delta_objective_error_threshold_palm)

        if residual_on_right:
            lst_S_in = op_S_factors.get_list_of_factors()[:nb_factors_so_far]
            f_lambda, lst_S_out, _, objective_palm_fine, _ = \
                func_fine_tune_step_palm4msa(
                    lst_S_init=lst_S_in + [new_residual])
            for i in range(nb_factors_so_far):
                op_S_factors.set_factor(i, lst_S_out.get_factor(i))
            # TODO remove .toarray()?
            arr_residual = lst_S_out.get_factor(nb_factors_so_far).toarray()
        else:
            lst_S_in = op_S_factors.get_list_of_factors()[-nb_factors_so_far:]
            f_lambda, lst_S_out, _, objective_palm_fine, _ = \
                func_fine_tune_step_palm4msa(
                    lst_S_init=[new_residual] + lst_S_in)
            for i in range(nb_factors_so_far):
                op_S_factors.set_factor(-nb_factors_so_far + i, lst_S_out.get_factor(i + 1))
            # TODO remove .toarray()?
            arr_residual = lst_S_out.get_factor(0).toarray()

        lst_objective_split_fine_fac_k.append(objective_palm_fine)
        lst_objectives.append(tuple(lst_objective_split_fine_fac_k))

        if return_objective_function:
            objective_function[k, 2] = compute_objective_function(arr_X_target,
                                                                  f_lambda,
                                                                  op_S_factors)

    # last factor is residual of last palm4LED
    if residual_on_right:
        op_S_factors.set_factor(-1, arr_residual)
    else:
        op_S_factors.set_factor(0, arr_residual)

    if return_objective_function:
        objective_function[nb_factors - 1, :] = np.array(
            [compute_objective_function(arr_X_target, f_lambda, op_S_factors)] * 3)

    arr_X_curr = f_lambda * op_S_factors.compute_product()

    return f_lambda, op_S_factors, arr_X_curr, lst_objectives, objective_function


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
    raise DeprecationWarning("Use palm4msa_fast4 instead")

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
            L_norm, v0 = L.compute_spectral_norm() \
                if L.n_factors > 0 else 1, None
            R_norm, v0 = R.compute_spectral_norm() \
                if R.n_factors > 0 else 1, None
            min_c_value = (f_lambda * L_norm * R_norm) ** 2
            # add epsilon because it is exclusive minimum
            c = min_c_value * 1.001
            logger.debug("Lipsitchz constant value: {}; c value: {}"
                         .format(min_c_value, c))
            # compute new factor value
            Sj = update_S(S_factors_op.get_factor(j), L, R, c, f_lambda,
                          lst_projection_functions[j])
            S_factors_op.set_factor(j, Sj)

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
    raise DeprecationWarning("Use palm4msa_fast4 instead")

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
            L_norm, v0 = L.compute_spectral_norm() \
                if L.n_factors > 0 else 1, None
            R_norm, v0 = R.compute_spectral_norm() \
                if R.n_factors > 0 else 1, None
            min_c_value = (f_lambda * L_norm * R_norm) ** 2
            # add epsilon because it is exclusive minimum
            c = min_c_value * 1.001
            logger.debug(
                "Lipsitchz constant value: {}; c value: {}".format(min_c_value,
                                                                   c))
            # compute new factor value
            res = f_lambda * S_factors_op.compute_product() - arr_X_target
            res_RT = R.dot(res.T).T if R.n_factors > 0 else res
            LT_res_RT = L.transpose().dot(
                res_RT) if L.n_factors > 0 else res_RT
            grad_step = 1. / c * f_lambda * LT_res_RT

            Sj = S_factors_op.get_factor(j)

            # normalize because all factors must have norm 1
            S_proj = coo_matrix(lst_projection_functions[j](Sj - grad_step))
            S_proj /= np.sqrt(S_proj.power(2).sum())

            S_factors_op.set_factor(j, S_proj)

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


def palm4msa_fast3(arr_X_target: np.array,
                   lst_S_init: list,
                   nb_factors: int,
                   lst_projection_functions: list,
                   f_lambda_init: float,
                   nb_iter: int,
                   update_right_to_left=True,
                   graphical_display=False,
                   debug=False,
                   track_objective=False):
    """
    lst S init contains factors in decreasing indexes (e.g: the order along which they are multiplied in the product).
        example: S5 S4 S3 S2 S1

    lst S [-j] = Sj

    """
    raise DeprecationWarning("Use palm4msa_fast4 instead")

    if debug:
        logger.debug('Norme de arr_X_target: {}'.format(
            np.linalg.norm(arr_X_target, ord='fro')))
    # initialization
    f_lambda = f_lambda_init
    S_factors_op = SparseFactors(lst_S_init)

    assert np.all(S_factors_op.shape == arr_X_target.shape)
    assert S_factors_op.n_factors > 0
    assert S_factors_op.n_factors == nb_factors

    if track_objective:
        objective_function = np.zeros((nb_iter, nb_factors + 1))

    if update_right_to_left:
        # range arguments: start, stop, step
        factor_number_generator = range(-1, -(nb_factors + 1), -1)
    else:
        factor_number_generator = range(0, nb_factors, 1)
    # main loop
    i_iter = 0
    delta_objective_error_threshold = 1e-6
    delta_objective_error = np.inf
    obj_fun_prev = None
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
            # print(nb_factors, L.n_factors+R.n_factors+1, L.n_factors,
            #       R.n_factors, j)

            # compute minimum c value (according to paper)
            L_norm, v0 = L.compute_spectral_norm() \
                if L.n_factors > 0 else 1, None
            R_norm, v0 = R.compute_spectral_norm() \
                if R.n_factors > 0 else 1, None
            min_c_value = (f_lambda * L_norm * R_norm) ** 2
            # add epsilon because it is exclusive minimum
            c = min_c_value * 1.001
            if debug:
                logger.debug("Lipsitchz constant value: {}; c value: {}"
                             .format(min_c_value, c))
            # compute new factor value
            res = f_lambda * S_factors_op.compute_product() - arr_X_target
            res_RT = R.dot(res.T).T if R.n_factors > 0 else res
            LT_res_RT = \
                L.transpose().dot(res_RT) if L.n_factors > 0 else res_RT
            grad_step = 1. / c * f_lambda * LT_res_RT

            Sj = S_factors_op.get_factor(j)

            # normalize because all factors must have norm 1
            S_proj = coo_matrix(lst_projection_functions[j](Sj - grad_step))
            S_proj /= np.sqrt(S_proj.power(2).sum())

            S_factors_op.set_factor(j, S_proj)

            if track_objective:
                objective_function[i_iter, j - 1] = \
                    compute_objective_function(arr_X_target,
                                               _f_lambda=f_lambda,
                                               _lst_S=S_factors_op)
                # print(objective_function[i_iter, j - 1])

        # re-compute the full factorisation
        arr_X_curr = S_factors_op.compute_product()
        # update lambda
        f_lambda = update_scaling_factor(X=arr_X_target, X_est=arr_X_curr)
        # print(f_lambda)

        if debug:
            logger.debug("Lambda value: {}".format(f_lambda))

        if track_objective:
            objective_function[i_iter, -1] = \
                compute_objective_function(arr_X_target, _f_lambda=f_lambda,
                                           _lst_S=S_factors_op)

        if debug and track_objective:
            logger.debug("Iteration {}; Objective value: {}"
                         .format(i_iter, objective_function[i_iter, -1]))

        obj_fun = np.linalg.norm(arr_X_target - f_lambda * arr_X_curr,
                                 ord='fro') ** 2
        if i_iter >= 1:
            delta_objective_error = \
                np.abs(obj_fun - obj_fun_prev) / obj_fun_prev
        obj_fun_prev = obj_fun
        # TODO vérifier que l'erreur absolue est plus petite que le
        # threshold plusieurs fois d'affilée

        i_iter += 1
    if track_objective:
        objective_function = objective_function[:i_iter, :]
    else:
        objective_function = None

    if graphical_display and track_objective:
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


def palm4msa_fast4(arr_X_target: np.array,
                   lst_S_init: list,
                   nb_factors: int,
                   lst_projection_functions: list,
                   f_lambda_init: float,
                   nb_iter: int,
                   update_right_to_left=True,
                   track_objective=False,
                   delta_objective_error_threshold=1e-6):
    """
    lst S init contains factors in decreasing indexes (e.g: the order along which they are multiplied in the product).
        example: S5 S4 S3 S2 S1

    lst S [-j] = Sj

    :param arr_X_target:
    :param lst_S_init:
    :param nb_factors:
    :param lst_projection_functions:
    :param f_lambda_init:
    :param nb_iter:
    :param update_right_to_left:
    :param track_objective: If true, the objective function is computed for each factor and not only at the end of each iteration.
    :param delta_objective_error_threshold:
    :return:
    """
    logger.debug('Norme de arr_X_target: {}'.format(np.linalg.norm(arr_X_target, ord='fro')))
    # initialization
    f_lambda = f_lambda_init
    S_factors_op = SparseFactors(lst_S_init)

    assert np.all(S_factors_op.shape == arr_X_target.shape)
    assert S_factors_op.n_factors > 0
    assert S_factors_op.n_factors == nb_factors

    if track_objective:
        objective_function = np.ones((nb_iter, nb_factors + 1)) * -1  # (nb_factors + 1) because of the lambda
    else:
        objective_function =  np.ones((nb_iter, 1)) * -1

    if update_right_to_left:
        # range arguments: start, stop, step
        factor_number_generator = range(-1, -(nb_factors + 1), -1)
    else:
        factor_number_generator = range(0, nb_factors, 1)
    # main loop
    i_iter = 0
    delta_objective_error = np.inf

    init_vectors_norm_comp_L = [None] * nb_factors
    init_vectors_norm_comp_R = [None] * nb_factors

    while ((i_iter < nb_iter) and (delta_objective_error > delta_objective_error_threshold)):

        for machine_idx_fac, j in enumerate(factor_number_generator):
            if lst_projection_functions[j].__name__ == "constant_proj":
                if track_objective:
                    objective_function[i_iter, machine_idx_fac] = compute_objective_function(arr_X_target,
                                                                                             _f_lambda=f_lambda,
                                                                                             _lst_S=S_factors_op)
                    logger.debug("Iteration {}; Factor idx {}; Objective value {}".format(i_iter, j, objective_function[i_iter, machine_idx_fac]))
                continue

            L = S_factors_op.get_L(j)
            R = S_factors_op.get_R(- j - 1)
            # R = S_factors_op.get_R(nb_factors - j - 1)
            # print(nb_factors, L.n_factors+R.n_factors+1, L.n_factors,
            #       R.n_factors, j, -j-1)

            # compute minimum c value (according to paper)
            L_norm, init_vectors_norm_comp_L[j] = L.compute_spectral_norm(init_vector_eigs_v0=init_vectors_norm_comp_L[j]) \
                if L.n_factors > 0 else (1, init_vectors_norm_comp_L[j])
            R_norm, init_vectors_norm_comp_R[j] = R.compute_spectral_norm(init_vector_eigs_v0=init_vectors_norm_comp_R[j]) \
                if R.n_factors > 0 else (1, init_vectors_norm_comp_R[j])
            min_c_value = (f_lambda * L_norm * R_norm) ** 2  # lipsitchz constant
            # add epsilon because it is exclusive minimum
            c = min_c_value * 1.001
            logger.debug("Lipsitchz constant value: {}; c value: {}"
                         .format(min_c_value, c))
            # compute new factor value
            # todo check if it is not redundant to recompute the S_factors_op
            res = f_lambda * S_factors_op.compute_product() - arr_X_target
            # res_RH = R.dot(res.T).T if R.n_factors > 0 else res
            res_RH = S_factors_op.apply_RH(n_factors=-j-1, X=res)
            # res_RH = S_factors_op.apply_RH(n_factors=nb_factors-j-1, X=res)
            LH_res_RH = S_factors_op.apply_LH(n_factors=j, X=res_RH)
            grad_step = 1. / c * f_lambda * LH_res_RH

            Sj = S_factors_op.get_factor(j)

            # normalize because all factors must have norm 1
            # todo verifier que ce qui est fait ici n'est pas sous-optimal
            S_proj = lst_projection_functions[j](Sj - grad_step)
            S_proj = coo_matrix(S_proj)
            S_proj /= np.sqrt(S_proj.power(2).sum())

            S_factors_op.set_factor(j, S_proj)

            if track_objective:
                objective_function[i_iter, machine_idx_fac] = compute_objective_function(arr_X_target,
                                               _f_lambda=f_lambda,
                                               _lst_S=S_factors_op)
                logger.debug("Iteration {}; Factor idx {}; Objective value {}".format(i_iter, j, objective_function[i_iter, machine_idx_fac]))


        # re-compute the full factorisation
        # todo check if it is not redundant to recompute the S_factors_op
        arr_X_curr = S_factors_op.compute_product()

        # update lambda
        f_lambda = update_scaling_factor(X=arr_X_target, X_est=arr_X_curr)

        logger.debug("Lambda value: {}".format(f_lambda))

        objective_function[i_iter, -1] = \
            compute_objective_function(arr_X_target, _f_lambda=f_lambda,
                                       _lst_S=S_factors_op)

        logger.debug("Iteration {}; Objective value: {}"
                     .format(i_iter, objective_function[i_iter, -1]))

        if i_iter >= 1:
            delta_objective_error = np.abs(objective_function[i_iter, -1] - objective_function[i_iter-1, -1]) / objective_function[i_iter-1, -1]
            logger.debug("Delta objective: {}".format(delta_objective_error))

        # TODO vérifier que l'erreur absolue est plus petite que le threshold plusieurs fois d'affilée

        i_iter += 1

    return f_lambda, S_factors_op, arr_X_curr, objective_function, i_iter


palm4msa = palm4msa_fast4

if __name__ == '__main__':
    from scipy.linalg import hadamard
    from pyqalm.utils import get_lambda_proxsplincol

    do_hierarchical = True

    if not do_hierarchical:
        data = dict()
        # data['hadamard'] = hadamard(64)

        data['rando'] = datasets.make_blobs(64, 128)[0]
        data['rando'] = data['rando'] / np.linalg.norm(data['rando'])

        # n_rows = 64
        # n_cols = 77
        # X = np.random.randn(n_rows, n_cols)
        # data['random matrix'] = X

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
            nb_iter = 20
            update_right_to_left = True
            # f_lambda_ref, lst_S_ref, arr_X_curr_ref, objective_function_ref, \
            # i_iter_ref = \
            #     palm4msa_slow(X,
            #              lst_S_init=lst_S_init,
            #              nb_factors=nb_factors,
            #              lst_projection_functions=lst_projection_functions,
            #              f_lambda_init=f_lambda_init,
            #              nb_iter=nb_iter,
            #              update_right_to_left=update_right_to_left,
            #              graphical_display=graphical_display)

            out = palm4msa_fast4(X,
                                 lst_S_init=lst_S_init,
                                 nb_factors=nb_factors,
                                 lst_projection_functions=lst_projection_functions,
                                 # lst_projection_functions=lst_projection_functions_fast,
                                 f_lambda_init=f_lambda_init,
                                 nb_iter=nb_iter,
                                 update_right_to_left=update_right_to_left,
                                 track_objective=True)

            f_lambda, S_factors_op, arr_X_curr, objective_function, i_iter = out
            objective_function = objective_function[:, -1][objective_function[:, -1] != -1]
            plt.plot(objective_function / np.linalg.norm(data['rando'])**2)
            plt.show()

            plt.semilogy(np.abs(objective_function[np.arange(1, len(objective_function))] - objective_function[np.arange(1, len(objective_function))-1]) / objective_function[np.arange(1, len(objective_function))-1])
            plt.show()
    else:
        data = dict()
        data['hadamard'] = hadamard(32)

        n_rows = 64
        n_cols = 77
        X = np.random.randn(n_rows, n_cols)
        data['random matrix'] = X

        X = data['hadamard']
        X = data['random matrix']
        d = np.min(X.shape)
        if X.shape[1] == d:
            X = X.T
        nb_factors = int(np.log2(d))

        nb_iter = 300

        # lst_factors = [np.eye(d) for _ in range(nb_factors)]
        # lst_factors[-1] = np.zeros((d, d))  # VE
        lst_factors = []
        for _ in range(nb_factors - 1):
            lst_factors.append(np.eye(d))
        lst_factors.append(np.zeros(X.shape))
        _lambda = 1.

        lst_proj_op_by_fac_step = []
        nb_keep_values = 2 * d
        for k in range(nb_factors - 1):
            nb_values_residual = int(d / 2 ** (k + 1)) * d
            dct_step_lst_nb_keep_values = {
                "split": [get_lambda_proxsplincol(nb_keep_values),
                          get_lambda_proxsplincol(nb_values_residual)],
                "finetune": [get_lambda_proxsplincol(nb_keep_values)] * (
                        k + 1) + [
                                get_lambda_proxsplincol(
                                    nb_values_residual)]
            }
            lst_proj_op_by_fac_step.append(dct_step_lst_nb_keep_values)

        # out0 = hierarchical_palm4msa_slow(
        #     arr_X_target=X,
        #     lst_S_init=lst_factors,
        #     lst_dct_projection_function=lst_proj_op_by_fac_step,
        #     f_lambda_init=_lambda,
        #     nb_iter=nb_iter,
        #     update_right_to_left=True,
        #     residual_on_right=True,
        #     graphical_display=False)

        hierarchical_palm4msa_fast = hierarchical_palm4msa
        import daiquiri
        import logging

        daiquiri.setup(level=logging.INFO)

        out1 = hierarchical_palm4msa_fast(
            arr_X_target=X,
            lst_S_init=lst_factors,
            lst_dct_projection_function=lst_proj_op_by_fac_step,
            f_lambda_init=_lambda,
            nb_iter=nb_iter,
            update_right_to_left=True,
            residual_on_right=True,
            return_objective_function=True)
