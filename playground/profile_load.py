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
    for i in range(n_examples//batch_size):
        u @ x[:, i*batch_size:(i+1)*batch_size]


def run_mmap(filename, n_examples, batch_size, data_dim, n_projections):
    u = np.random.randn(n_projections, data_dim)
    x = np.load(filename + '.npy', mmap_mode='r')
    for i in range(n_examples // batch_size):
        u @ x[:, i * batch_size:(i + 1) * batch_size]


def run_one_file_per_batch(filename, n_examples, batch_size, data_dim,
                           n_projections):
    u = np.random.randn(n_projections, data_dim)
    for i in range(n_examples // batch_size):
        x = np.load(filename + '_' + str(i) + '.npy', mmap_mode='r')
        u @ x


def main():
    try:
        mkdir('data')
    except FileExistsError:
        pass
    params = {
        'filename': 'data/tmp',
        'batch_size': 2 ** 8,
        'n_examples': 2 ** 14,
        'data_dim': 2 ** 9,
        'n_projections': 2 ** 10,
    }
    create_data(**params)
    run_data_in_ram(**params)
    run_mmap(**params)
    run_one_file_per_batch(**params)


if __name__ == '__main__':

    import line_profiler

    lp = line_profiler.LineProfiler()
    lp.add_function(run_data_in_ram)
    lp.add_function(run_mmap)
    lp.add_function(run_one_file_per_batch)
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()

    stats_file = 'profile_load.lprof'
    lp.dump_stats(stats_file)
