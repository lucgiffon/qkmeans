# -*- coding: utf-8 -*-
"""
Just run ```kernprof -l tuto_line_profiling_noninvasive.py```

.. moduleauthor:: Valentin Emiya
"""
import numpy as np
from scipy.sparse import csr_matrix
import atexit


def f(n):
    A = np.eye(n)
    Asp = csr_matrix(A)
    x = np.random.randn(n, n)
    Y = A @ x
    Ysp = Asp @ x
    return Y, Ysp


if __name__ == '__main__':
    import line_profiler

    profile = line_profiler.LineProfiler()
    atexit.register(profile.print_stats)
    profile.add_function(f)
    for p in range(8, 12):
        f(2**p)
