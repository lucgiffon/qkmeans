# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
from time import process_time, perf_counter
import numpy as np
import matplotlib.pyplot as plt

from yafe.base import Experiment

from pyqalm.data_structures import SparseFactors, create_sparse_factors


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


class Solver:
    def __init__(self, sparsity_level):
        self.sparsity_level = sparsity_level

    def __call__(self, vector):
        axis_size = vector.shape[0]
        sparse_factors = create_sparse_factors(
            axis_size=axis_size,
            n_factors=int(np.ceil(np.log2(axis_size))),
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


def measure(solution_data, solved_data, task_params=None, source_data=None,
            problem_data=None):
    return {'Elapsed time PT': solved_data['Elapsed time PT'],
            'Elapsed time PC': solved_data['Elapsed time PC'],
            }


n_seeds = 10


class SparsityTimeExperiment(Experiment):
    def __init__(self, force_reset=True):
        Experiment.__init__(self,
                            name='Running time for varing sparsity',
                            get_data=get_data,
                            get_problem=Problem,
                            get_solver=Solver,
                            measure=measure,
                            force_reset=force_reset,
                            # data_path=temp_data_path,
                            log_to_file=False,
                            log_to_console=False
                            )

    def add_tasks(self,
                  data_params={'size': 2 ** np.arange(6, 13, 2),
                               'seed': np.arange(n_seeds, dtype=int)},
                  problem_params={'no_param': [0]},
                  solver_params={'sparsity_level': [2, 4, 8, 16, 32]}):
        Experiment.add_tasks(self,
                             data_params=data_params,
                             problem_params=problem_params,
                             solver_params=solver_params)

    def plot_results(self):
        results = self.load_results(array_type='xarray')
        results_pt = results.sel(problem_no_param=0, measure='Elapsed time PT')
        results_pt = results_pt.mean('data_seed')
        for sparsity in results_pt.solver_sparsity_level.values:
            plt.semilogx(results_pt.data_size,
                         results_pt.sel(solver_sparsity_level=sparsity),
                         label='Sparsity level {}'.format(sparsity))

        results_pc = results.sel(problem_no_param=0, measure='Elapsed time PC')
        results_pc = results_pc.mean('data_seed')
        for sparsity in results_pc.solver_sparsity_level.values:
            plt.loglog(results_pc.data_size,
                       results_pc.sel(solver_sparsity_level=sparsity),
                       '--',
                       label='Sparsity level {}'.format(sparsity))

        plt.xlabel('Size')
        plt.ylabel('Average running time (s)')
        plt.title('Matrix-vector fast product')
        plt.grid()
        plt.legend()


if __name__ == '__main__':
    run_all = input('Run all? (Y/N)')
    if run_all == 'Y':
        exp = SparsityTimeExperiment()
        exp.display_status()
        exp.add_tasks()
        exp.display_status()
        exp.generate_tasks()
        exp.display_status()
        exp.launch_experiment()
        exp.display_status()
        exp.collect_results()
        exp.plot_results()
        plt.savefig(exp.name)
        plt.show()
    else:
        exp = SparsityTimeExperiment(force_reset=False)
        exp.plot_results()
        plt.savefig(exp.name)
        plt.show()
