# -*- coding: utf-8 -*-
"""
Just run ```kernprof -l tuto_line_profiling_noninvasive.py```

.. moduleauthor:: Valentin Emiya
"""
import numpy as np
from scipy.sparse import csr_matrix


def f(n):
    A = np.eye(n)
    Asp = csr_matrix(A)
    x = np.random.randn(n, n)
    Y = A @ x
    Ysp = Asp @ x
    return Y, Ysp


def main():
    for p in range(8, 12):
        f(2 ** p)


if __name__ == '__main__':
    import line_profiler
    import atexit

    lp = line_profiler.LineProfiler()
    # atexit.register(profile.print_stats)
    lp.add_function(f)
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()

    stats_file = 'tuto_ni.lprof'
    lp.dump_stats(stats_file)
    # from pyprof2calltree import convert, visualize
    # convert(stats_file, 'out.kgrind')
    # visualize(stats_file)
