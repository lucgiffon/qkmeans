from sklearn.metrics.pairwise import rbf_kernel

from qkmeans.FWHT import FWHT
import numpy as np
from sklearn import datasets
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from sklearn.preprocessing import StandardScaler
from qkmeans.kernel.kernel import prepare_nystrom, nystrom_transformation
from qkmeans.utils import compute_euristic_gamma


class UfastOperator(LinearOperator):
    """
    Will be used as a set of landmark points for the nystrom approximation.
    """
    def __init__(self, seeds_vec, fast_op=FWHT):
        self.fast_op = fast_op
        self.seeds = seeds_vec # represents the diagonals: \in R^{s x d}
        super().__init__(shape=(self.seeds.shape[0] * self.seeds.shape[1], self.seeds.shape[1]), dtype=self.seeds.dtype)

    def _matvec(self, x):
        # \in R^{sd}
        seed_products = self.seeds * x # represents all the vector obtained after multiplying the inputs by the diagonals: \in R^{s x d}
        result = np.vstack([self.fast_op(v) for v in seed_products])
        return result

    def toarray(self):
        # \in R^{sd x d}
        result = np.hstack([v[:, np.newaxis] * hadamard(v.size).T for v in self.seeds]).T
        return result

if __name__ == "__main__":
    x, y = datasets.load_digits(return_X_y=True)
    # x = np.pad(x, (0, 2), 'constant')

    nb_seeds = 5
    dim = 2
    data = x
    scaler = StandardScaler(with_std=False)
    data = scaler.fit_transform(data)

    gamma = compute_euristic_gamma(data)
    # data_norm = np.linalg.norm(x, axis=1)[:, np.newaxis]
    data_norm = None
    plt.scatter(data[:, 0], data[:, 1], color="b")
    seeds = data[np.random.permutation(data.shape[0])[:nb_seeds]]
    uop = UfastOperator(seeds, FWHT)
    uop_arr = uop.toarray()
    uniform_sample = data[np.random.permutation(data.shape[0])[:uop_arr.shape[0]]]
    uop_arr_norm = np.linalg.norm(uop_arr, axis=1)[:, np.newaxis]
    plt.scatter(seeds[:, 0], seeds[:, 1], marker="x", s=200,  color='r')
    plt.scatter(uop_arr[:, 0], uop_arr[:, 1], color='g')
    # plt.show()

    real_kernel_value = rbf_kernel(x, gamma=gamma)

    # nystrom with uop
    metric = prepare_nystrom(uop_arr, None, gamma=gamma)
    nystrom_embedding = nystrom_transformation(x, uop_arr, metric, None, None, gamma=gamma)
    nystrom_approx_kernel_value = nystrom_embedding @ nystrom_embedding.T
    print("uop", np.linalg.norm(nystrom_approx_kernel_value -  real_kernel_value) / np.linalg.norm(real_kernel_value))

    # nystrom with uniform sample
    metric = prepare_nystrom(uniform_sample, None, gamma=gamma)
    nystrom_embedding = nystrom_transformation(x, uniform_sample, metric, None, None, gamma=gamma)
    nystrom_approx_kernel_value_uniform = nystrom_embedding @ nystrom_embedding.T
    print("uniform ful_size", np.linalg.norm(nystrom_approx_kernel_value_uniform -  real_kernel_value) / np.linalg.norm(real_kernel_value))

    # nystrom with seeds
    metric = prepare_nystrom(seeds, None, gamma=gamma)
    nystrom_embedding = nystrom_transformation(x, seeds, metric, None, None, gamma=gamma)
    nystrom_approx_kernel_value_seeds = nystrom_embedding @ nystrom_embedding.T
    print("seed small size", np.linalg.norm(nystrom_approx_kernel_value_seeds -  real_kernel_value) / np.linalg.norm(real_kernel_value))


    # print((uop @ np.random.rand(dim)).size)
    #
    # a = np.random.rand(16)
    # print(FWHT(a))