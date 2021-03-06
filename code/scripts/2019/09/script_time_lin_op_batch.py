# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
from time import process_time, perf_counter
import numpy as np
import matplotlib.pyplot as plt

from yafe.base import Experiment

from qkmeans.data_structures import SparseFactors, create_sparse_factors


def get_data(size, seed, n_vectors):
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
        if self.sparsity_level is None:
            sparse_factors = np.random.randn(axis_size, axis_size)
        else:
            sparse_factors = create_sparse_factors(
                shape=(axis_size, axis_size),
                n_factors=int(np.ceil(np.log2(axis_size))),
                sparsity_level=self.sparsity_level)

        t0 = process_time()
        _ = sparse_factors @ vector
        elapsed_time = process_time() - t0

        # t0_pc = perf_counter()
        # _ = sparse_factors @ vector
        # elapsed_time_pc = perf_counter() - t0_pc

        return {'Elapsed time PT': elapsed_time,
                'Elapsed time PC': None}

    def __str__(self):
        return 'Solver with sparsity level {}'.format(self.sparsity_level)


def measure(solution_data, solved_data, task_params=None, source_data=None,
            problem_data=None):
    return {'Elapsed time PT': solved_data['Elapsed time PT'],
            'Elapsed time PC': solved_data['Elapsed time PC'],
            }


n_seeds = 5


class RunningTime4LinOpBatchExperiment(Experiment):
    def __init__(self, force_reset=True):
        Experiment.__init__(self,
                            name='RunningTime4LinOpBatch',
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
                               'seed': np.arange(n_seeds, dtype=int),
                               'n_vectors': [1, 100, 500]},
                  problem_params={'no_param': [0]},
                  solver_params={'sparsity_level': [None, 2, 4]}):
        Experiment.add_tasks(self,
                             data_params=data_params,
                             problem_params=problem_params,
                             solver_params=solver_params)

    def plot_results(self, with_area=True, fontsize=24, figsize=(14, 8),
                     linewidth=3):
        import matplotlib as mpl
        mpl.rcParams['figure.figsize'] = figsize
        mpl.rcParams['lines.linewidth'] = linewidth
        mpl.rcParams['font.size'] = fontsize

        plt.figure()
        results = self.load_results(array_type='xarray')
        results_pt = results.sel(problem_no_param=0, measure='Elapsed time PT')
        results_pt_mean = results_pt.mean('data_seed')
        results_pt_min = results_pt.min('data_seed')
        results_pt_max = results_pt.max('data_seed')
        for n_vectors in results_pt.data_n_vectors.values:
            for sparsity in results_pt.solver_sparsity_level.values:
                if sparsity is None:
                    line = plt.loglog(
                        results_pt_mean.data_size,
                        results_pt_mean.sel(solver_sparsity_level=sparsity,
                                            data_n_vectors=n_vectors)
                        / n_vectors,
                        label='Dense, {} vec'.format(n_vectors))
                else:
                    line = plt.loglog(
                        results_pt_mean.data_size,
                        results_pt_mean.sel(solver_sparsity_level=sparsity,
                                            data_n_vectors=n_vectors)
                        / n_vectors,
                        label='Sparsity {}, {} vec'
                            .format(sparsity, n_vectors))
                if with_area:
                    plt.fill_between(
                        results_pt.data_size,
                        results_pt_min.sel(solver_sparsity_level=sparsity,
                                           data_n_vectors=n_vectors)
                        / n_vectors,
                        results_pt_max.sel(solver_sparsity_level=sparsity,
                                           data_n_vectors=n_vectors)
                        / n_vectors,
                        alpha=0.5,
                        facecolor=line[0].get_color(),
                        linestyle='None')

        # results_pc = results.sel(problem_no_param=0, measure='Elapsed time PC')
        # results_pc = results_pc.mean('data_seed')
        # for sparsity in results_pc.solver_sparsity_level.values:
        #     plt.loglog(results_pc.data_size,
        #                results_pc.sel(solver_sparsity_level=sparsity),
        #                '--',
        #                label='Sparsity level {}'.format(sparsity))

        plt.xlabel('Data size $D$')
        plt.ylabel('Average running time (s)')
        # plt.title('Matrix-vector product')
        plt.grid()
        plt.legend()


exp = RunningTime4LinOpBatchExperiment(force_reset=False)
if __name__ == '__main__':
    from yafe.utils import generate_oar_script

    answer = input('1-Create experiment\n2-Run all\n3-Plot\n4-Run a job\n5 '
                   'Re-generate script\n')
    if answer == '1':
        exp = RunningTime4LinOpBatchExperiment(force_reset=True)
        exp.display_status()
        exp.add_tasks()
        exp.display_status()
        exp.generate_tasks()
        exp.display_status()
        generate_oar_script(script_file_path=__file__,
                            xp_var_name='exp',
                            batch_size=10,
                            oar_walltime='00:30:00',
                            activate_env_command='source activate py36'
                            )
    elif answer == '2':
        exp = RunningTime4LinOpBatchExperiment(force_reset=False)
        exp.display_status()
        exp.launch_experiment()
        exp.display_status()
        exp.collect_results()
        exp.plot_results()
        plt.savefig(exp.name)
        plt.show()
    elif answer == '3':
        exp = RunningTime4LinOpBatchExperiment(force_reset=False)

        exp.plot_results(with_area=False)
        plt.savefig(exp.name)

        exp.plot_results(with_area=True)
        plt.savefig(exp.name + '_with_area')
        plt.show()
    elif answer == '4':
        exp = RunningTime4LinOpBatchExperiment(force_reset=False)
        exp.launch_experiment(task_ids=[0])
    elif answer == '5':
        exp.display_status()
        batch_size = int(input('Batch size?'))
        generate_oar_script(script_file_path=__file__,
                            xp_var_name='exp',
                            batch_size=batch_size,
                            oar_walltime='00:20:00',
                            activate_env_command='source activate py36'
                            )
