import unittest
import numpy as np
from qkmeans.kernel.kernel import special_rbf_kernel, prepare_nystrom, nystrom_transformation
from qkmeans.data_structures import create_sparse_factors
from qkmeans.core.utils import get_squared_froebenius_norm_line_wise
from qkmeans.utils import compute_euristic_gamma, mnist_dataset, fashion_mnist_dataset
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.extmath import row_norms

import logging
from qkmeans.utils import logger
logger.setLevel(logging.INFO)


class TestKernel(unittest.TestCase):
    def setUp(self):
        self.n_features = 2000
        self.n_data = 100
        sparsity = 2

        self.sparse_data = create_sparse_factors(
                shape=(self.n_data*2, self.n_features),
                n_factors=int(np.ceil(np.log2(min(self.n_data*2, self.n_features)))),
                sparsity_level=sparsity)
        self.data = self.sparse_data.compute_product(return_array=True)

        self.data_verylittle = np.random.rand(*self.data.shape) * 1e2

        self.data_norm = get_squared_froebenius_norm_line_wise(self.data)
        self.sparse_data_norm = get_squared_froebenius_norm_line_wise(self.sparse_data)
        self.data_norm_verylittle = get_squared_froebenius_norm_line_wise(self.data_verylittle)

        self.gamma = compute_euristic_gamma(self.data)
        self.random_data = np.random.rand(self.n_data, self.n_features)
        self.mnist = mnist_dataset()["x_train"].astype(np.float64)
        self.fashionmnist = fashion_mnist_dataset()["x_train"].astype(np.float64)
        # self.caltech = caltech_dataset(28)["x_train"].astype(np.float64)


        self.pairs_data = {
            "notSparse": self.data[:self.n_data],
            "mnist": self.mnist[:self.n_data],
            # "caltech": self.caltech[:self.n_data],
            "fashmnist": self.fashionmnist[:self.n_data],
        }
        self.example_data = {
            "notSparse": self.data[self.n_data:self.n_data*2],
            "mnist": self.mnist[self.n_data:self.n_data*2],
            # "caltech": self.caltech[self.n_data:self.n_data*2],
            "fashmnist": self.fashionmnist[self.n_data:self.n_data*2]
        }

        self.norm_data = {
            "notSparse": row_norms(self.pairs_data["notSparse"], squared=True)[:, np.newaxis],
            "mnist": get_squared_froebenius_norm_line_wise(self.pairs_data["mnist"])[:, np.newaxis],
            # "caltech": row_norms(self.pairs_data["caltech"], squared=True)[:, np.newaxis],
            "fashmnist": row_norms(self.pairs_data["fashmnist"], squared=True)[:, np.newaxis],
        }

        self.norm_example_data = {
            "notSparse": row_norms(self.example_data["notSparse"], squared=True)[:, np.newaxis],
            "mnist": get_squared_froebenius_norm_line_wise(self.example_data["mnist"]),
            # "caltech": row_norms(self.example_data["caltech"], squared=True)[:, np.newaxis],
            "fashmnist": row_norms(self.example_data["fashmnist"], squared=True)[:, np.newaxis],
        }

        self.gamma_data = {
            "notSparse": compute_euristic_gamma(self.pairs_data["notSparse"]),
            "mnist": compute_euristic_gamma(self.pairs_data["mnist"]),
            # "caltech": compute_euristic_gamma(self.pairs_data["caltech"]),
            "fashmnist": compute_euristic_gamma(self.pairs_data["fashmnist"]),
        }

        # create data
        # create sparse data

    def test_kernel(self):
        # compute kernel with special rbf kernel
        # compute kernel with sklearn kernel
        # compute kernel between sparse_data_ and sparse_data
        # compute_kernel between sparse_data and data
        # compute kernel between sparse_data and random_data
        # compute_kernel between data and random_data

        # sklearn_kernel_first = rbf_kernel(self.data, self.data, self.gamma)
        # sklearn_kernel_verylittle = rbf_kernel(self.data_verylittle, self.data_verylittle)
        for name_pair, pair in self.pairs_data.items():
            data_norm = self.norm_data[name_pair]
            gamma = self.gamma_data[name_pair]
            sklearn_kernel = rbf_kernel(pair, pair, gamma=gamma)

            special_kernel = special_rbf_kernel(pair, pair, gamma=gamma, norm_X=data_norm, norm_Y=data_norm.T, exp_outside=False)
            special_kernel_flag = special_rbf_kernel(pair, pair, gamma=gamma, norm_X=data_norm, norm_Y=data_norm.T, exp_outside=True)
            special_kernel[special_kernel <1e-12] = 0
            special_kernel_flag[special_kernel_flag <1e-12] = 0
            sklearn_kernel[sklearn_kernel < 1e-12] = 0

            equality = np.allclose(sklearn_kernel, special_kernel)
            equality_flag = np.allclose(sklearn_kernel, special_kernel_flag)

            delta = np.linalg.norm(special_kernel - sklearn_kernel)
            delta_flag = np.linalg.norm(special_kernel_flag - sklearn_kernel)
            print("Delta flag: {}; delta: {}".format(delta_flag, delta))

            self.assertTrue(delta_flag < delta)
            self.assertTrue(equality, msg=name_pair)
            self.assertTrue(equality_flag, msg=name_pair)

    def test_norm(self):
        self.assertTrue((self.data_norm == self.sparse_data_norm).all())

    def test_nystrom(self):

        for name_pair, pair in self.pairs_data.items():
            gamma = self.gamma_data[name_pair]
            data_norm = self.norm_data[name_pair]

            print(name_pair)
            sklearn_nystrom = Nystroem(gamma=gamma, n_components=self.n_data)
            sklearn_transformation = sklearn_nystrom.fit_transform(pair)
            # sklearn_transformation_example = sklearn_nystrom.transform(example_data[name_pair])

            special_metric = prepare_nystrom(pair, data_norm, gamma=gamma)
            special_transformation = nystrom_transformation(pair, pair, special_metric, data_norm, data_norm.T, gamma)
            # special_transformation_example = nystrom_transformation(example_data[name_pair], pair, special_metric, data_norm, norm_example_data[name_pair], gamma)

            sklearn_mat = sklearn_transformation @ sklearn_transformation.T
            special_mat = special_transformation @ special_transformation.T


            equality = np.allclose(sklearn_mat, special_mat)
            try:
                self.assertTrue(equality, msg="Sklearn nystrom approximatio is different for data {}".format(name_pair))
            except Exception as e:
                real_matrix = rbf_kernel(pair, gamma=gamma)

                delta_sk_real = np.linalg.norm(sklearn_mat - real_matrix)
                delta_spec_real = np.linalg.norm(special_mat - real_matrix)

                if not np.allclose(delta_sk_real, delta_spec_real):
                    raise Exception("Exception 1: {}. Exception 2: They are not even equally distant from real matrix".format(str(e)))
                raise e

            # sklearn_mat_example = sklearn_transformation_example @ sklearn_transformation_example.T
            # special_mat_example = special_transformation_example @ special_transformation_example.T

            # equality = np.allclose(sklearn_mat_example, special_mat_example)
            # self.assertTrue(equality, msg="example " + name_pair)



if __name__ == '__main__':
    unittest.main()
