from pyfaust import wht
from pyfaust.fact import hierarchical as hierarchical_faust
from pyfaust.factparams import ParamsHierarchical, ConstraintList, StoppingCriterion
from pyfaust.proj import splincol
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import daiquiri
import pprint
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.data_structures import SparseFactors
from qkmeans.utils import logger, visual_evaluation_palm4msa
from numpy.linalg import norm

from qkmeans.palm.palm_fast import hierarchical_palm4msa
from qkmeans.core.utils import build_constraint_set_smart


def pyqalm_hierarchical_hadamard(H, d):
    # Parameters for palm
    nb_iter = 30
    nb_factors = int(np.log2(d))
    sparsity_factor = 2

    # Create init sparse factors as identity (the first sparse matrix will remain constant)
    lst_factors = [np.eye(d) for _ in range(nb_factors)]
    lst_factors[-1] = np.zeros((d, d))
    _lambda = 1.  # init the scaling factor at 1

    # Create the projection operators for each factor
    lst_proj_op_by_fac_step, lst_proj_op_by_fac_step_desc = build_constraint_set_smart(left_dim=d,
                                                                                       right_dim=d,
                                                                                       nb_factors=nb_factors,
                                                                                       sparsity_factor=sparsity_factor,
                                                                                       residual_on_right=True,
                                                                                       fast_unstable_proj=False, constant_first=False)

    # Call the algorithm
    final_lambda, final_factors, final_X, _, _ = hierarchical_palm4msa(
        arr_X_target=H,
        lst_S_init=lst_factors,
        lst_dct_projection_function=lst_proj_op_by_fac_step,
        f_lambda_init=_lambda,
        nb_iter=nb_iter,
        update_right_to_left=True,
        residual_on_right=True)

    return final_lambda, lst_factors, final_factors, final_X

def main_compare_hierarchical():
    daiquiri.setup(level=logging.INFO)
    min_power2 = 2
    # max_power2 = 9
    max_power2 = 6
    dims = np.array([int(2 ** i) for i in range(min_power2, max_power2)])
    nb_replicates = 3
    times_pyqalm_hierarchical = np.empty((len(dims), nb_replicates))
    times_faust_hierarchical = np.empty((len(dims), nb_replicates))
    times_pyqalm_mat = np.empty((len(dims), nb_replicates))
    times_faust_mat = np.empty((len(dims), nb_replicates))
    times_pyqalm_vec = np.empty((len(dims), nb_replicates))
    times_faust_vec = np.empty((len(dims), nb_replicates))
    mse_pyqalm = np.empty((len(dims), nb_replicates))
    mse_faust = np.empty((len(dims), nb_replicates))
    diffs_mat = np.empty((len(dims), nb_replicates))
    for i_dim, dim in enumerate(dims):
        for i_seed in range(nb_replicates):
            FH = wht(dim)
            H = FH.toarray()
            rand_matrix_of_examples = np.random.randn(dim, 1000)

            ### Hierarchical multiplication

            start_pyqalm_hierarchical = time.time()
            final_lambda, lst_factors, final_factors, final_X = pyqalm_hierarchical_hadamard(H, dim)
            stop_pyqalm_hierarchical = time.time()
            # visual_evaluation_palm4msa(H, lst_factors, final_factors, final_X)
            F_pyqalm = final_factors * final_lambda
            time_pyqalm_hierarchical = stop_pyqalm_hierarchical - start_pyqalm_hierarchical
            times_pyqalm_hierarchical[i_dim, i_seed] = time_pyqalm_hierarchical
            diff_pyqalm = np.linalg.norm(H - final_X) / (dim**2)
            mse_pyqalm[i_dim, i_seed] = diff_pyqalm

            start_faust_hierarchical = time.time()
            F_faust = hierarchical_faust(H, 'squaremat')
            stop_faust_hierarchical = time.time()
            diff_faust = np.linalg.norm(H - F_faust.toarray()) / (dim**2)
            mse_faust[i_dim, i_seed] = diff_faust

            time_faust_hierarchical = stop_faust_hierarchical - start_faust_hierarchical
            times_faust_hierarchical[i_dim, i_seed] = time_faust_hierarchical

            ### Operator/matrix multiplication

            start_pyqalm_mat = time.time()
            r_pyqalm = F_pyqalm @ rand_matrix_of_examples
            stop_pyqalm_mat = time.time()
            time_pyqalm_mat = stop_pyqalm_mat - start_pyqalm_mat
            times_pyqalm_mat[i_dim, i_seed] = time_pyqalm_mat

            start_faust_mat = time.time()
            r_faust = F_faust * rand_matrix_of_examples
            stop_faust_mat = time.time()
            time_faust_mat = stop_faust_mat - start_faust_mat
            times_faust_mat[i_dim, i_seed] = time_faust_mat

            diff_mat = np.linalg.norm(r_pyqalm - r_faust) / (r_pyqalm.shape[0] * r_pyqalm.shape[1])
            diffs_mat[i_dim, i_seed] = diff_mat
            ### Operator / vector multiplication

            n_vec = 10
            start_pyqalm_vec = time.time()
            for i in range(n_vec):
                r = F_pyqalm @ rand_matrix_of_examples[:, i]
            stop_pyqalm_vec = time.time()
            time_pyqalm_vec = stop_pyqalm_vec - start_pyqalm_vec
            times_pyqalm_vec[i_dim, i_seed] = time_pyqalm_vec

            start_faust_vec = time.time()
            for i in range(n_vec):
                r = F_faust * rand_matrix_of_examples[:, i]
            stop_faust_vec = time.time()
            time_faust_vec = stop_faust_vec - start_faust_vec
            times_faust_vec[i_dim, i_seed] = time_faust_vec


    np.savez("results_time_hierarchical", faust=times_faust_hierarchical, pyqalm=times_pyqalm_hierarchical)
    np.savez("results_time_mat", faust=times_faust_mat, pyqalm=times_pyqalm_mat)
    np.savez("results_time_vec", faust=times_faust_vec, pyqalm=times_pyqalm_vec)
    np.savez("results_mse", faust=mse_faust, pyqalm=mse_pyqalm)
    np.save("results_diff_mat", diffs_mat)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
            x=dims,
            y=np.mean(times_faust_hierarchical, axis=1),
            name="faust",
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=np.std(times_faust_hierarchical, axis=1),
                visible=True)
        ))

    fig.add_trace(go.Scatter(
            x=dims,
            y=np.mean(times_pyqalm_hierarchical, axis=1),
            name="pyqalm",
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=np.std(times_pyqalm_hierarchical, axis=1),
                visible=True)
        ))
    fig.update_layout(title="hierarchical ")

    fig.update_xaxes(title_text='Dimension')
    fig.update_yaxes(title_text='Time')
    fig.show()
    fig.write_image("hierarchical.png")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=dims,
            y=np.mean(times_faust_mat, axis=1),
            name="faust",
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=np.std(times_faust_mat, axis=1),
                visible=True)
        ))
    fig.add_trace(go.Scatter(
            x=dims,
            y=np.mean(times_pyqalm_mat, axis=1),
            name="pyqalm",
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=np.std(times_pyqalm_mat, axis=1),
                visible=True)
        ))
    fig.update_layout(title="mat")

    fig.update_xaxes(title_text='Dimension')
    fig.update_yaxes(title_text='Time')
    fig.show()
    fig.write_image("mat.png")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=dims,
            y=np.mean(times_faust_vec, axis=1),
            name="faust",
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=np.std(times_faust_vec, axis=1),
                visible=True)
        ))

    fig.add_trace(go.Scatter(
            x=dims,
            y=np.mean(times_pyqalm_vec, axis=1),
            name="pyqalm",
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=np.std(times_pyqalm_vec, axis=1),
                visible=True)
        ))
    fig.update_layout(title="vec")
    fig.update_xaxes(title_text='Dimension')
    fig.update_yaxes(title_text='Time')
    fig.show()
    fig.write_image("vec.png")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=dims,
            y=np.mean(mse_faust, axis=1),
            name="faust",
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=np.std(mse_faust, axis=1),
                visible=True)
        ))

    fig.add_trace(go.Scatter(
            x=dims,
            y=np.mean(mse_pyqalm, axis=1),
            name="pyqalm",
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=np.std(mse_pyqalm, axis=1),
                visible=True)
        ))
    fig.update_layout(title="mse")
    fig.update_xaxes(title_text='Dimension')
    fig.update_yaxes(title_text='MSE')
    fig.show()
    fig.write_image("mse.png")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=dims,
            y=np.mean(diffs_mat, axis=1),
            name="faust",
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=np.std(diffs_mat, axis=1),
                visible=True)
        ))

    fig.update_layout(title="Diff matmul")
    fig.update_xaxes(title_text='Dimension')
    fig.update_yaxes(title_text='Mean square diff')
    fig.show()
    fig.write_image("diffmatmul.png")

def main_compare_prod_vec():
    daiquiri.setup(level=logging.INFO)
    min_power2 = 2
    max_power2 = 18
    # max_power2 = 6
    n_vec = 100

    dims = np.array([int(2 ** i) for i in range(min_power2, max_power2)])
    nb_replicates = 3
    times_pyqalm_vec = np.empty((len(dims), nb_replicates))
    times_faust_vec = np.empty((len(dims), nb_replicates))
    for i_dim, dim in enumerate(dims):
        for i_seed in range(nb_replicates):
            F_faust = wht(dim)
            F_pyqalm = SparseFactors([F_faust.factors(i) for i in range(F_faust.numfactors())])

            rand_matrix_of_examples = np.random.randn(dim, n_vec)

            start_pyqalm_vec = time.time()
            for i in range(n_vec):
                r = F_pyqalm @ rand_matrix_of_examples[:, i]
            stop_pyqalm_vec = time.time()
            time_pyqalm_vec = stop_pyqalm_vec - start_pyqalm_vec
            times_pyqalm_vec[i_dim, i_seed] = time_pyqalm_vec

            start_faust_vec = time.time()
            for i in range(n_vec):
                r = F_faust @ rand_matrix_of_examples[:, i]
            stop_faust_vec = time.time()
            time_faust_vec = stop_faust_vec - start_faust_vec
            times_faust_vec[i_dim, i_seed] = time_faust_vec

    np.savez("results_time_vec_bis", faust=times_faust_vec, pyqalm=times_pyqalm_vec)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dims,
        y=np.mean(times_faust_vec, axis=1),
        name="faust",
        error_y=dict(
            type='data',  # value of error bar given in data coordinates
            array=np.std(times_faust_vec, axis=1),
            visible=True)
    ))

    fig.add_trace(go.Scatter(
        x=dims,
        y=np.mean(times_pyqalm_vec, axis=1),
        name="pyqalm",
        error_y=dict(
            type='data',  # value of error bar given in data coordinates
            array=np.std(times_pyqalm_vec, axis=1),
            visible=True)
    ))
    fig.update_layout(title="vec")

    fig.update_xaxes(title_text='Dimension')
    fig.update_yaxes(title_text='Time')
    fig.show()
    fig.write_image("vec_bis.png")


if __name__ == "__main__":
    main_compare_prod_vec()