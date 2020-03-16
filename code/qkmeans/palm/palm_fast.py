# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
.. moduleauthor:: Luc Giffon
"""
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from qkmeans.palm.utils import compute_objective_function, update_scaling_factor
from scipy.sparse import csr_matrix

from qkmeans.utils import logger
from qkmeans.data_structures import SparseFactors
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
    assert len(lst_dct_projection_function) == nb_factors - 1, "Number of factor {} and number of constraints {} are different".format(len(lst_dct_projection_function), nb_factors-1)
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

    :param arr_X_target: The target to approximate.
    :param lst_S_init: The initial list of sparse factors.
    :param nb_factors: The number of factors.
    :param lst_projection_functions: The projection function for each of the sparse factor.
    :param f_lambda_init: The initial scaling factor.
    :param nb_iter: The number of iteration before stopping.
    :param update_right_to_left: Tells the algorithm to update factors from right to left (S1 first)
    :param graphical_display: Make a graphical representation of results.
    :param track_objective: If true, the objective function is computed for each factor and not only at the end of each iteration.
    :param delta_objective_error_threshold: The normalized difference threshold between error at two successive iterations threshold below which the computation is stopped.

    :return: the sparse factorization but careful: the final X isn't multiplyed by lambda
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
            S_proj = lst_projection_functions[j](Sj - grad_step)
            S_proj = csr_matrix(S_proj)
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
    from qkmeans.utils import get_lambda_proxsplincol

    do_hierarchical = False

    if not do_hierarchical:
        data = dict()
        # data['hadamard'] = hadamard(64)

        data['rando'] = datasets.make_blobs(64, 128)[0]
        data['rando'] = data['rando'] / np.linalg.norm(data['rando'])


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
                [get_lambda_proxsplincol(nb_keep_values, fast_unstable=True)] * nb_factors \
                + [get_lambda_proxsplincol(nb_values_residual, fast_unstable=True)]

            f_lambda_init = 1
            nb_iter = 20
            update_right_to_left = True

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
                "split": [get_lambda_proxsplincol(nb_keep_values, fast_unstable=True),
                          get_lambda_proxsplincol(nb_values_residual, fast_unstable=True)],
                "finetune": [get_lambda_proxsplincol(nb_keep_values, fast_unstable=True)] * (
                        k + 1) + [
                                get_lambda_proxsplincol(
                                    nb_values_residual, fast_unstable=True)]
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
