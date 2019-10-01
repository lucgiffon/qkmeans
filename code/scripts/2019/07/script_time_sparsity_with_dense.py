# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
from time import process_time, perf_counter
import numpy as np
import matplotlib.pyplot as plt

from yafe.base import Experiment

from qkmeans.data_structures import SparseFactors, create_sparse_factors


def get_data(size, seed):
    n_vectors = 100
    return {'vector': np.random.RandomState(seed).randn(size, n_vectors)}


class Problem:
    def __init__(self, no_param):
        self.no_param = no_param

    def __call__(self, vector):
        problem_data = {'vector': vector}
        solution_data = dict()
        return problem_data, solution_data

    def __str__(self):
        return 'Problem with size {} and seed {}'.format(self.size, self.seed)


class SparsySolver:
    def __init__(self, sparsity_level, nb_lin):
        self.sparsity_level = sparsity_level
        self.nb_lin = nb_lin

    def __call__(self, vector):
        nb_col = vector.shape[0]
        nb_lin = self.nb_lin

        sparse_factors = create_sparse_factors(
            (nb_lin, nb_col),
            # axis_size=axis_size,
            n_factors=int(np.ceil(np.log2(min(nb_lin, nb_col)))),
            sparsity_level=self.sparsity_level)

        t0 = process_time()
        _ = sparse_factors @ vector
        elapsed_time = process_time() - t0

        t0_pc = perf_counter()
        _ = sparse_factors @ vector
        elapsed_time_pc = perf_counter() - t0_pc

        return {'Elapsed time PT': elapsed_time,
                'Elapsed time PC': elapsed_time_pc}

    def __str__(self):
        return 'Solver with sparsity level {}'.format(self.sparsity_level)


class DenseSolver:
    def __init__(self, nb_lin):
        self.nb_lin = nb_lin

    def __call__(self, vector):
        nb_col = vector.shape[0]
        nb_lin = self.nb_lin

        dense_matrix = np.random.rand(nb_lin, nb_col)

        t0 = process_time()
        _ = dense_matrix @ vector
        elapsed_time = process_time() - t0

        t0_pc = perf_counter()
        _ = dense_matrix @ vector
        elapsed_time_pc = perf_counter() - t0_pc

        return {'Elapsed time PT': elapsed_time,
                'Elapsed time PC': elapsed_time_pc}

    def __str__(self):
        return 'Dense solver'


def measure(solution_data, solved_data, task_params=None, source_data=None,
            problem_data=None):
    return {'Elapsed time PT': solved_data['Elapsed time PT'],
            'Elapsed time PC': solved_data['Elapsed time PC'],
            }


n_seeds = 2


def plot_results(sparsy_exp, dense_exp):
    sparsy_results = sparsy_exp.load_results(array_type='xarray')
    sparsy_results_pt = sparsy_results.sel(problem_no_param=0, measure='Elapsed time PT')
    sparsy_results_pt = sparsy_results_pt.mean('data_seed')

    dense_results = dense_exp.load_results(array_type='xarray')
    dense_results_pt = dense_results.sel(problem_no_param=0, measure='Elapsed time PT')
    dense_results_pt = dense_results_pt.mean('data_seed')




    for sparsity in sparsy_results_pt.solver_sparsity_level.values:
        f, ax = plt.subplots()
        results_sparsy = sparsy_results_pt.sel(solver_sparsity_level=sparsity)
        for nb_lin in results_sparsy.solver_nb_lin.values:
            plt.semilogx(results_sparsy.data_size,
                         results_sparsy.sel(solver_nb_lin=nb_lin),
                         label='Sparsity level {} - Nb row {}'.format(sparsity, nb_lin))

        for nb_lin in dense_results_pt.solver_nb_lin.values:
            plt.semilogx(dense_results_pt.data_size,
                         dense_results_pt.sel(solver_nb_lin=nb_lin), "--",
                         label='Dense - Nb row {}'.format(nb_lin))


    # results_pc = results.sel(problem_no_param=0, measure='Elapsed time PC')
    # results_pc = results_pc.mean('data_seed')
    # for sparsity in results_pc.solver_sparsity_level.values:
    #     plt.loglog(results_pc.data_size,
    #                results_pc.sel(solver_sparsity_level=sparsity),
    #                '--',
    #                label='Sparsity level {}'.format(sparsity))

        plt.xlabel('Size')
        plt.ylabel('Average running time (s)')
        plt.title('Matrix-vector fast product')
        plt.grid()
        plt.legend()
        plt.savefig("Run_time_sparsity_{}".format(sparsity))
        plt.show()


def run_sparsy_experiment():
    sparsy_experiment = Experiment(
        name='Running time for varying sparsity',
        get_data=get_data,
        get_problem=Problem,
        get_solver=SparsySolver,
        measure=measure,
        force_reset=False,
        # data_path=temp_data_path,
        log_to_file=False,
        log_to_console=False
    )
    sparsy_experiment.display_status()
    sparsy_experiment.add_tasks(
        data_params={
            'size': 2 ** np.arange(6, 13, 2),
            'seed': np.arange(n_seeds, dtype=int)},
        problem_params={'no_param': [0]},
        solver_params={
            'sparsity_level': [2, 4, 8],
            "nb_lin": [10, 16, 30, 50, 100]})
    sparsy_experiment.display_status()
    sparsy_experiment.generate_tasks()
    sparsy_experiment.display_status()
    sparsy_experiment.launch_experiment()
    sparsy_experiment.display_status()
    sparsy_experiment.collect_results()
    return sparsy_experiment


def run_dense_experiment():
    dense_experiment = Experiment(
        name='Running time for dense',
        get_data=get_data,
        get_problem=Problem,
        get_solver=DenseSolver,
        measure=measure,
        force_reset=False,
        # data_path=temp_data_path,
        log_to_file=False,
        log_to_console=False
    )
    dense_experiment.display_status()
    dense_experiment.add_tasks(
        data_params={
            'size': 2 ** np.arange(6, 13, 2),
            'seed': np.arange(n_seeds, dtype=int)},
        problem_params={'no_param': [0]},
        solver_params={
            "nb_lin": [10, 16, 30, 50, 100]})
    dense_experiment.display_status()
    dense_experiment.generate_tasks()
    dense_experiment.display_status()
    dense_experiment.launch_experiment()
    dense_experiment.display_status()
    dense_experiment.collect_results()
    return dense_experiment

if __name__ == '__main__':
    sparsy_experiment_result = run_sparsy_experiment()
    dense_experiment_result = run_dense_experiment()

    plot_results(sparsy_experiment_result, dense_experiment_result)