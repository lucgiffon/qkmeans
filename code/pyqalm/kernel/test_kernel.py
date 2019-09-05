import unittest
import numpy as np
from pyqalm.kernel.kernel import special_rbf_kernel, prepare_nystrom, nystrom_transformation
from pyqalm.data_structures import create_sparse_factors
from pyqalm.qk_means.utils import get_squared_froebenius_norm_line_wise
from pyqalm.utils import compute_euristic_gamma, mnist_dataset, fashion_mnist_dataset, caltech_dataset
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.n_features = 2000
        self.n_data = 100
        sparsity = 2

        self.sparse_data = create_sparse_factors(
                shape=(self.n_data, self.n_features),
                n_factors=int(np.ceil(np.log2(min(self.n_data, self.n_features)))),
                sparsity_level=sparsity)
        self.data = self.sparse_data.compute_product(return_array=True)

        self.data_verylittle = np.random.rand(*self.data.shape) * 1e2

        self.data_norm = get_squared_froebenius_norm_line_wise(self.data)
        self.sparse_data_norm = get_squared_froebenius_norm_line_wise(self.sparse_data)
        self.data_norm_verylittle = get_squared_froebenius_norm_line_wise(self.data_verylittle)

        self.gamma = compute_euristic_gamma(self.data)
        self.random_data = np.random.rand(self.n_data, self.n_features)
        self.mnist = mnist_dataset()["x_train"]
        self.fashionmnist = fashion_mnist_dataset()["x_train"]
        self.caltech = caltech_dataset(28)["x_train"]
        # create data
        # create sparse data

    def test_kernel(self):
        # compute kernel with special rbf kernel
        # compute kernel with sklearn kernel
        # compute kernel between sparse_data_ and sparse_data
        # compute_kernel between sparse_data and data
        # compute kernel between sparse_data and random_data
        # compute_kernel between data and random_data

        pairs_data = {
            # "Sparse-Sparse": (self.sparse_data, self.sparse_data),
            # "Sparse-notSparse": (self.sparse_data, self.data),
            # "notSparse-Sparse": (self.data, self.sparse_data),
            # "notSparse-notSparse": (self.data, self.data),
            "verylittle": (self.data_verylittle, self.data_verylittle),
        }
        sklearn_kernel_first = rbf_kernel(self.data, self.data, self.gamma)
        sklearn_kernel_verylittle = rbf_kernel(self.data_verylittle, self.data_verylittle)
        for name_pair, pair in pairs_data.items():
            if name_pair == "verylittle":
                sklearn_kernel = sklearn_kernel_verylittle
                data_norm = self.data_norm_verylittle
            else:
                sklearn_kernel = sklearn_kernel_first
                data_norm = self.data_norm

            special_kernel = special_rbf_kernel(pair[0], pair[1], self.gamma, data_norm, data_norm, exp_outside=False)
            special_kernel_flag = special_rbf_kernel(pair[0], pair[1], self.gamma, data_norm, data_norm, exp_outside=True)
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
        pairs_data = {
            # "notSparse": self.data[:self.n_data],
            "mnist": self.mnist[:self.n_data],
            # "caltech": self.caltech[:self.n_data],
            # "fashmnist": self.fashionmnist[:self.n_data],
        }
        example_data = {
            # "notSparse": self.data[self.n_data:self.n_data*2],
            "mnist": self.mnist[self.n_data:self.n_data*2],
            # "caltech": self.caltech[self.n_data:self.n_data*2],
            # "fashmnist": self.fashionmnist[self.n_data:self.n_data*2]
        }

        norm_data = {
            # "notSparse": get_squared_froebenius_norm_line_wise(pairs_data["notSparse"]),
            "mnist": get_squared_froebenius_norm_line_wise(pairs_data["mnist"]),
            # "caltech": get_squared_froebenius_norm_line_wise(pairs_data["caltech"]),
            # "fashmnist": get_squared_froebenius_norm_line_wise(pairs_data["fashmnist"]),
        }


        norm_example_data = {
            # "notSparse": get_squared_froebenius_norm_line_wise(example_data["notSparse"]),
            "mnist": get_squared_froebenius_norm_line_wise(example_data["mnist"]),
            # "caltech": get_squared_froebenius_norm_line_wise(example_data["caltech"]),
            # "fashmnist": get_squared_froebenius_norm_line_wise(example_data["fashmnist"]),
        }

        gamma_data = {
            # "notSparse": compute_euristic_gamma(pairs_data["notSparse"]),
            "mnist": compute_euristic_gamma(pairs_data["mnist"]),
            # "caltech": compute_euristic_gamma(pairs_data["caltech"]),
            # "fashmnist": compute_euristic_gamma(pairs_data["fashmnist"]),
        }

        for name_pair, pair in pairs_data.items():
            gamma = gamma_data[name_pair]
            print(name_pair)
            sklearn_nystrom = Nystroem(gamma=gamma, n_components=self.n_data)
            sklearn_transformation = sklearn_nystrom.fit_transform(pair)
            sklearn_transformation_example = sklearn_nystrom.transform(example_data[name_pair])

            special_metric = prepare_nystrom(pair, norm_data[name_pair], gamma=gamma)
            special_transformation = nystrom_transformation(pair, pair, special_metric, norm_data[name_pair], norm_data[name_pair], gamma)
            special_transformation_example = nystrom_transformation(example_data[name_pair], pair, special_metric, norm_data[name_pair], norm_example_data[name_pair], gamma)

            sklearn_mat = sklearn_transformation @ sklearn_transformation.T
            special_mat = special_transformation @ special_transformation.T

            equality = np.allclose(sklearn_mat, special_mat)
            self.assertTrue(equality, msg=name_pair)

            sklearn_mat_example = sklearn_transformation_example @ sklearn_transformation_example.T
            special_mat_example = special_transformation_example @ special_transformation_example.T

            equality = np.allclose(sklearn_mat_example, special_mat_example)
            self.assertTrue(equality, msg="example " + name_pair)

if __name__ == '__main__':
    unittest.main()
