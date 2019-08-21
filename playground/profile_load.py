# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""
import numpy as np
from os import mkdir


def create_data(filename, n_examples, batch_size, data_dim, n_projections):
    x = np.random.randn(data_dim, n_examples)
    np.save(file=filename + '.npy', arr=x)
    for i in range(n_examples//batch_size):
        np.save(file=filename + '_' + str(i) + '.npy',
                arr=x[:, i*batch_size:(i+1)*batch_size])


def run_data_in_ram(filename, n_examples, batch_size, data_dim, n_projections):
    u = np.random.randn(n_projections, data_dim)
    x = np.load(filename + '.npy')
    s = 0
    for i in range(n_examples//batch_size):
        xi = x[:, i*batch_size:(i+1)*batch_size]
        xi = np.copy(xi)
        y = u @ xi
        s += np.sum(y)
    return s


def run_mmap(filename, n_examples, batch_size, data_dim, n_projections):
    u = np.random.randn(n_projections, data_dim)
    x = np.load(filename + '.npy', mmap_mode='r')
    s = 0
    for i in range(n_examples // batch_size):
        xi = x[:, i * batch_size:(i + 1) * batch_size]
        xi = np.copy(xi)
        y = u @ xi
        s += np.sum(y)
    return s


def run_mmap_rand(filename, n_examples, batch_size, data_dim, n_projections):
    u = np.random.randn(n_projections, data_dim)
    x = np.load(filename + '.npy', mmap_mode='r')
    s = 0
    for _ in range(n_examples // batch_size):
        i = np.random.randint(0, n_examples // batch_size)
        xi = x[:, i * batch_size:(i + 1) * batch_size]
        xi = np.copy(xi)
        y = u @ xi
        s += np.sum(y)
    return s


def run_one_file_per_batch(filename, n_examples, batch_size, data_dim,
                           n_projections):
    u = np.random.randn(n_projections, data_dim)
    s = 0
    for i in range(n_examples // batch_size):
        xi = np.load(filename + '_' + str(i) + '.npy', mmap_mode='r')
        xi = np.copy(xi)
        y = u @ xi
        s += np.sum(y)
    return s


def main(params):
    print('run_data_in_ram')
    run_data_in_ram(**params)
    print('run_mmap')
    run_mmap(**params)
    print('run_mmap_rand')
    run_mmap_rand(**params)
    print('run_one_file_per_batch')
    run_one_file_per_batch(**params)


if __name__ == '__main__':

    import line_profiler

    params = {
        'filename': 'data/tmp',
        'batch_size': 2 ** 11,
        'n_examples': 2 ** 18,
        'data_dim': 2 ** 9,
        'n_projections': 2 ** 11,
    }
    try:
        mkdir('data')
    except FileExistsError:
        pass
    print('Create data')
    create_data(**params)
    lp = line_profiler.LineProfiler()
    lp.add_function(run_data_in_ram)
    lp.add_function(run_mmap)
    lp.add_function(run_mmap_rand)
    lp.add_function(run_one_file_per_batch)
    lp_wrapper = lp(main)
    lp_wrapper(params)
    lp.print_stats(output_unit=1e-3)

    stats_file = 'profile_load.lprof'
    lp.dump_stats(stats_file)
    print('Run the following command to display the results:')
    print('$ python -m line_profiler {}'.format(stats_file))
