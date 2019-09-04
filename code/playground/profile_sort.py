# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
import numpy as np
from os import mkdir


def main(n_runs=20):
    n = 2**11
    for i in range(n_runs):
        x = np.random.randn(n, n)
        np.sort(x, axis=0)
        np.argsort(x, axis=0)
        np.argsort(x, axis=1)
        np.argsort(x.T, axis=0)
        np.argsort(x.T, axis=1)
        np.argsort(-x, axis=0)
        np.argsort(-x, axis=1)

        k = 2
        np.partition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=1)
        np.argpartition(x.T, kth=k, axis=0)
        np.argpartition(x.T, kth=k, axis=1)
        np.argpartition(-x, kth=k, axis=0)
        np.argpartition(-x, kth=k, axis=1)

        k = 4
        np.partition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=1)
        np.argpartition(x.T, kth=k, axis=0)
        np.argpartition(x.T, kth=k, axis=1)
        np.argpartition(-x, kth=k, axis=0)
        np.argpartition(-x, kth=k, axis=1)

        k = 8
        np.partition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=1)
        np.argpartition(x.T, kth=k, axis=0)
        np.argpartition(x.T, kth=k, axis=1)
        np.argpartition(-x, kth=k, axis=0)
        np.argpartition(-x, kth=k, axis=1)

        k = 32
        np.partition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=1)
        np.argpartition(x.T, kth=k, axis=0)
        np.argpartition(x.T, kth=k, axis=1)
        np.argpartition(-x, kth=k, axis=0)
        np.argpartition(-x, kth=k, axis=1)

        k = n - 2
        np.partition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=1)
        np.argpartition(x.T, kth=k, axis=0)
        np.argpartition(x.T, kth=k, axis=1)
        np.argpartition(-x, kth=k, axis=0)
        np.argpartition(-x, kth=k, axis=1)

        k = n - 4
        np.partition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=1)
        np.argpartition(x.T, kth=k, axis=0)
        np.argpartition(x.T, kth=k, axis=1)
        np.argpartition(-x, kth=k, axis=0)
        np.argpartition(-x, kth=k, axis=1)

        k = n - 8
        np.partition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=1)
        np.argpartition(x.T, kth=k, axis=0)
        np.argpartition(x.T, kth=k, axis=1)
        np.argpartition(-x, kth=k, axis=0)
        np.argpartition(-x, kth=k, axis=1)

        k = n - 32
        np.partition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=0)
        np.argpartition(x, kth=k, axis=1)
        np.argpartition(x.T, kth=k, axis=0)
        np.argpartition(x.T, kth=k, axis=1)
        np.argpartition(-x, kth=k, axis=0)
        np.argpartition(-x, kth=k, axis=1)


if __name__ == '__main__':

    import line_profiler

    # my_params = {
    #     'filename': 'data/tmp',
    #     'batch_size': 2 ** 8,
    #     'n_examples': 2 ** 7,
    #     'data_dim': 2 ** 5,
    #     'n_projections': 2 ** 7,
    # }
    try:
        mkdir('data')
    except FileExistsError:
        pass
    # rm_data(my_params['filename'])
    # create_data(**my_params)
    lp = line_profiler.LineProfiler()
    # lp.add_function(run_data_in_ram)
    # lp.add_function(run_mmap)
    # lp.add_function(run_mmap_rand)
    # lp.add_function(run_one_file_per_batch)
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats(output_unit=1e-3)

    stats_file = 'profile_sort.lprof'
    lp.dump_stats(stats_file)
    print('Run the following command to display the results:')
    print('$ python -m line_profiler {}'.format(stats_file))
    # rm_data(my_params['filename'])
