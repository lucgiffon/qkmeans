import numpy as np
import matplotlib.pyplot as plt

from pyqalm.utils import get_side_prod
from pyqalm.projection_operators import projection_operator


def test_get_side_prod():
    # TODO refactor into a unittest
    raise NotImplementedError
    nb_factors = 3
    d = 32
    nb_keep_values =64
    factors = [projection_operator(np.random.rand(d, d), nb_keep_values) for _ in range(nb_factors)]
    result = get_side_prod(factors)
    # truth =
    visual_evaluation_palm4msa()


def visual_evaluation_palm4msa(target, init_factors, final_factors, result):
    nb_factors = len(init_factors)
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 2, 1)
    plt.title("Objective")
    plt.imshow(target)
    plt.colorbar()
    plt.subplot(3, 2, 2)
    plt.title("Result (reconstruction)")
    plt.imshow(result)
    plt.colorbar()
    for i in range(nb_factors):
        plt.subplot(3, nb_factors, nb_factors + (i+1))
        plt.title("$\mathbf{S}_" + str(nb_factors-i) + "^{final}$")
        plt.imshow(final_factors[i])
        plt.subplot(3, nb_factors, nb_factors + nb_factors + (i+1))
        plt.title("$\mathbf{S}_" + str(nb_factors - i) + "^{init}$")
        plt.imshow(init_factors[i])
    plt.show()


