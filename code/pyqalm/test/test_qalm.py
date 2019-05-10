import numpy as np
import matplotlib.pyplot as plt

from pyqalm.qalm import projection_operator, get_side_prod


def test_get_side_prod():
    # TODO refactor into a unittest
    nb_factors = 3
    d = 32
    nb_keep_values =64
    factors = [projection_operator(np.random.rand(d, d), nb_keep_values) for _ in range(nb_factors)]
    result = get_side_prod(factors)
    # truth =
    visual_evaluation_palm4msa()


def visual_evaluation_palm4msa(target, init_factors, final_factors, result):
    nb_factors = len(init_factors)
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 2, 1)
    plt.imshow(target)
    plt.colorbar()
    plt.subplot(3, 2, 2)
    plt.imshow(result)
    plt.colorbar()
    print("Première ligne: Objectif \t | \t Résultat")
    print("Deuxième ligne: Les facteurs")
    print("Troisième ligne: Les facteurs initiaux")
    for i in range(nb_factors):
        plt.subplot(3, nb_factors, nb_factors + (i+1))
        plt.imshow(final_factors[i])
        plt.subplot(3, nb_factors, nb_factors + nb_factors + (i+1))
        plt.imshow(init_factors[i])
    plt.show()


