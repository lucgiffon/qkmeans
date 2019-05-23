import logging
import os
import random
import urllib

import pathlib
from copy import deepcopy

from pathlib import Path

from sklearn import datasets
from keras.datasets import mnist, fashion_mnist

from numpy import eye
import numpy as np
from numpy.linalg import multi_dot
import daiquiri
from pyqalm.palm.projection_operators import prox_splincol
from pyqalm import project_dir

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger("pyqalm")

def get_side_prod(lst_factors, id_shape=(0,0)):
    """
    Return the dot product between factors in lst_factors in order.

    exemple:

    lst_factors := [S1, S2, S3]
    return_value:= S1 @ S2 @ S3
    """
    # assert if the inner dimension of factors match: e.g. the multi dot product is feasible
    assert all([lst_factors[i].shape[-1] == lst_factors[i+1].shape[0] for i in range(len(lst_factors)-1)])

    if len(lst_factors) == 0:
        # convention from the paper itself: dot product of no factors equal Identity
        side_prod = eye(*id_shape)
    elif len(lst_factors) == 1:
        # if only 1 elm, return the elm itself (Identity * elm actually)
        side_prod = lst_factors[0]
    else:
        side_prod = multi_dot(lst_factors)
    return side_prod

def get_lambda_proxsplincol(nb_keep_values):
    return lambda mat: prox_splincol(mat, nb_keep_values)

def constant_proj(mat):
    raise NotImplementedError("This function should not be called but used for its name")

class ResultPrinter:
    """
    Class that handles 1-level dictionnaries and is able to print/write their values in a csv like format.
    """
    def __init__(self, *args, header=True, output_file=None):
        """
        :param args: the dictionnaries objects you want to print.
        :param header: tells if you want to print the header
        :param output_file: path to the outputfile. If None, no outputfile is written on ResultPrinter.print()
        """
        self.__dict = dict()
        self.__header = header
        self.__output_file = output_file

    def add(self, d):
        """
        Add dictionnary after initialisation.

        :param d: the dictionnary object you want to add.
        :return:
        """
        self.__dict.update(d)

    def _get_ordered_items(self):
        all_keys, all_values = zip(*self.__dict.items())
        arr_keys, arr_values = np.array(all_keys), np.array(all_values)
        indexes_sort = np.argsort(arr_keys)
        return list(arr_keys[indexes_sort]), list(arr_values[indexes_sort])

    def print(self):
        """
        Call this function whener you want to print/write to file the content of the dictionnaires.
        :return:
        """
        headers, values = self._get_ordered_items()
        headers = [str(h) for h in headers]
        s_headers = ",".join(headers)
        values = [str(v) for v in values]
        s_values = ",".join(values)
        if self.__header:
            print(s_headers)
        print(s_values)
        if self.__output_file is not None:
            with open(self.__output_file, "w+") as out_f:
                if self.__header:
                    out_f.write(s_headers + "\n")
                out_f.write(s_values + "\n")

def timeout_signal_handler(signum, frame):
    raise TimeoutError("More than 10 times slower than kmean")

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

class ObjectiveFunctionPrinter:
    def __init__(self, output_file:Path=None):
        self.objectives = dict()
        self.__output_file = output_file
        if output_file is not None and output_file.suffix != "":
            self.__ext_file = output_file.suffix
            self.__base_file = (output_file.parent / output_file.stem).as_posix()
        elif output_file is not None:
            self.__ext_file = ".csv"
            self.__base_file = self.__output_file.as_posix()

    def add(self, name, cols, arr):
        """
        arr must be either 1d or 2d arr
        :param name:
        :param arr:
        :return:
        """
        if len(arr.shape) > 2:
            raise ValueError("Trying to add a {}-D array to ObjectiveFunctionPrinter. Maximum dim accepted is 2.".format(len(arr.shape)))

        assert (len(arr.shape) == 1 and len(cols) == 1) or len(cols) == len(arr[0])

        self.objectives[name] = (cols, arr)

    def print(self):
        def print_2d_row(row):
            return ",".join(str(v) for v in row)

        def print_1d_row(row):
            return str(row)

        for name, (cols, arr) in self.objectives.items():
            cols_str = ",".join(cols)
            head = name + "\n" + "\n" + cols_str
            print(head)

            if self.__output_file is not None:
                path_arr = Path(self.__base_file + "_" + name + self.__ext_file)
                with open(path_arr, "w+") as out_f:
                    out_f.write(head + "\n")

            if len(arr.shape) == 1:
                print_row_fct = print_1d_row
            else:
                print_row_fct = print_2d_row
            for row in arr:
                str_row = print_row_fct(row)
                print(str_row)
                if self.__output_file is not None:
                    with open(path_arr, "a") as out_f:
                        out_f.write(str_row + "\n")

def get_random():
    val = str(random.randint(1, 10000000000))[1:8]
    # print(val)
    return val

class ParameterManager(dict):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--nb-cluster"] = int(self["--nb-cluster"])
        self["--nb-iteration"] = int(self["--nb-iteration"])

        self.__init_output_file()
        self.__init_seed()

    def __init_output_file(self):
        out_file = get_random()
        self["--output-file"] = out_file
        if out_file is not None and len(out_file.split(".")) > 1:
            raise ValueError("Output file name should be given without any extension (no `.` in the string)")
        if out_file is not None:
            self["--output-file_resprinter"] = Path(out_file + "_results.csv")
            self["--output-file_objprinter"] = Path(out_file + "_objective.csv")
            self["--output-file_centroidprinter"] = Path(out_file + "_centroids.npy")

    def __init_seed(self):
        if self["--seed"] is not None:
            np.random.seed(int(self["--seed"]))

    def get_dataset(self):
        """
        Return dataset in shape n x d.

        n: number of observations.
        d: dimensionality of observations.

        :return:
        """
        # todo normalize data before
        if self["--blobs"]:
            return blobs_dataset()
        elif self["--census"]:
            return census_dataset()
        elif self["--kddcup"]:
            return kddcup_dataset()
        elif self["--plants"]:
            return plants_dataset()
        elif self["--mnist"]:
            return mnist_dataset()
        elif self["--fashion-mnist"]:
            return fashion_mnist_dataset()
        else:
            raise NotImplementedError("Unknown dataset.")

    def get_initialization_centroids(self, input_data):
        """
        :param input_data: Matrix shape nxd (n: number of observations; d: dimensionality of observations)
        :return:
        """
        if self["--initialization"] == "random":
            return np.random.normal((self["--nb-cluster"], input_data.shape[1]))
        elif self["--initialization"] == "uniform_sampling":
            return input_data[np.random.permutation(input_data.shape[0])[:self["--nb-cluster"]]]
        else:
            raise NotImplementedError("Unknown initialization.")

def compute_euristic_gamma(dataset_full, slice_size=1000):
    """
    Given a dataset, return the gamma that should be used (euristically) when using a rbf kernel on this dataset.

    The formula: $\sigma^2 = 1/n^2 * \sum_{i, j}^{n}||x_i - x_j||^2$

    :param dataset: The dataset on which to look for the best sigma
    :return:
    """
    results = []
    # dataset_full = np.reshape(dataset_full, (-1, 1))
    if slice_size > dataset_full.shape[0]:
        slice_size = dataset_full.shape[0]
    for i in range(dataset_full.shape[0] // slice_size):
        if (i+1) * slice_size <= dataset_full.shape[0]:
            dataset = dataset_full[i * slice_size: (i+1) * slice_size]
            slice_size_tmp = slice_size
        else:
            dataset = dataset_full[i * slice_size:]
            slice_size_tmp = len(dataset)
        r1 = np.sum(dataset * dataset, axis=1)
        r1 = np.reshape(r1, [-1, 1])
        r2 = np.reshape(r1, [1, -1])
        d_mat = np.dot(dataset, dataset.T)
        d_mat = r1 - 2 * d_mat + r2
        results.append(1/slice_size_tmp**2 * np.sum(d_mat))
    return 1. / np.mean(results)

class ParameterManagerQmeans(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--sparsity-factor"] = int(self["--sparsity-factor"])

        self["--nb-iteration-palm"] = int(self["--nb-iteration-palm"])

        self.__init_nb_factors()

    def __init_nb_factors(self):
        if self["--nb-factors"] is not None:
            self["--nb-factors"] = int(self["--nb-factors"])


def blobs_dataset():
    blob_size = 500000
    blob_features = 2000
    blob_centers = 5000
    X, y = datasets.make_blobs(n_samples=blob_size, n_features=blob_features, centers=blob_centers)
    test_size = 1000
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    return {
        "x_train": X_train.reshape(X_train.shape[0], -1),
        # "y_train": y_train,
        # "x_test": X_test.reshape(X_test.shape[0], -1),
        # "y_test": y_test
    }

def census_dataset():
    data_dir = project_dir / "data/external" / "census.npz"
    loaded_npz = np.load(data_dir)
    return {"x_train": loaded_npz["x_train"]}

def kddcup_dataset():
    data_dir = project_dir / "data/external" / "kddcup.npz"
    loaded_npz = np.load(data_dir)
    return {"x_train": loaded_npz["x_train"]}

def plants_dataset():
    data_dir = project_dir / "data/external" / "plants.npz"
    loaded_npz = np.load(data_dir)
    return {"x_train": loaded_npz["x_train"]}

def mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return {
        "x_train": X_train.reshape(X_train.shape[0], -1),
        "y_train": y_train,
        "x_test": X_test.reshape(X_test.shape[0], -1),
        "y_test": y_test
    }

def fashion_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return {
        "x_train": X_train.reshape(X_train.shape[0], -1),
        "y_train": y_train,
        "x_test": X_test.reshape(X_test.shape[0], -1),
        "y_test": y_test
    }

def create_directory(_dir, parents=True, exist_ok=True):
    """
    Try to create the directory if it does not exist.

    :param dir: the path to the directory to be created
    :return: None
    """
    logger.debug("Creating directory {} if needed".format(_dir))
    pathlib.Path(_dir).mkdir(parents=parents, exist_ok=exist_ok)


def download_data(url, directory, name=None):
    """
    Download the file at the specified url

    :param url: the end-point url of the need file
    :type url: str
    :param directory: the target directory where to download the file
    :type directory: str
    :param name: the name of the target file downloaded
    :type name: str
    :return: The destination path of the downloaded file
    """
    create_directory(directory)
    logger.debug(f"Download file at {url}")
    if name is None:
        name = os.path.basename(os.path.normpath(url))
    s_file_path = os.path.join(directory, name)
    if not os.path.exists(s_file_path):
        urllib.request.urlretrieve(url, s_file_path)
        logger.debug("File {} has been downloaded to {}.".format(url, s_file_path))
    else:
        logger.debug("File {} already exists and doesn't need to be donwloaded".format(s_file_path))

    return s_file_path