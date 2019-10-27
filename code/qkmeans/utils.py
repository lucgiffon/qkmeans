"""
General utils functions for the project.
"""

import logging
import os
import random
import urllib

import pathlib

from pathlib import Path

import psutil
import scipy
from matplotlib import pyplot as plt
from sklearn import datasets
from keras.datasets import mnist, fashion_mnist

from numpy import eye
import numpy as np
from numpy.linalg import multi_dot
import daiquiri
from qkmeans.palm.projection_operators import prox_splincol
from qkmeans import project_dir
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import keras


daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger("pyqalm")


def log_memory_usage(context=None):
    """Logs current memory usage stats.
    See: https://stackoverflow.com/a/15495136

    :return: None
    """
    if context is not None:
        str_memory_usage = context + ":\t"
    else:
        str_memory_usage = ""

    PROCESS = psutil.Process(os.getpid())
    GIGA = 10 ** 9
    UNIT = "Go"
    # total, available, percent, used, free, _, _, _, _, _ = psutil.virtual_memory()
    process_v_mem = psutil.virtual_memory()
    total, available, used, free = process_v_mem.total / GIGA, process_v_mem.available / GIGA, process_v_mem.used / GIGA, process_v_mem.free / GIGA
    percent = used / total * 100
    proc = PROCESS.memory_info()[1] / GIGA
    str_memory_usage += 'process = {} {unit}; total = {} {unit}; available = {} {unit}; used = {} {unit}; free = {} {unit}; percent = {:.2f} %'.format(proc, total, available, used, free, percent, unit=UNIT)
    logger.debug(str_memory_usage)


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


def get_lambda_proxsplincol(nb_keep_values, fast_unstable=False):
    return lambda mat: prox_splincol(mat, nb_keep_values, fast_unstable=fast_unstable)


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
            self.__ext_file = ".npz"
            self.__base_file = self.__output_file.as_posix()

    def add(self, name, label_dims, arr):
        """
        arr must be either 1d or 2d arr
        :param name:
        :param arr:
        :return:
        """
        self.objectives[name] = (label_dims, arr)

    def print(self):
        def print_2d_row(row):
            return ",".join(str(v) for v in row)

        def print_1d_row(row):
            return str(row)

        np.savez(self.__output_file, **self.objectives)

        #
        # for name, (cols, arr) in self.objectives.items():
        #     cols_str = ",".join(cols)
        #     head = name + "\n" + "\n" + cols_str
        #     print(head)
        #
        #     if self.__output_file is not None:
        #         path_arr = Path(self.__base_file + "_" + name + self.__ext_file)
        #         with open(path_arr, "w+") as out_f:
        #             out_f.write(head + "\n")
        #
        #     if len(arr.shape) == 1:
        #         print_row_fct = print_1d_row
        #     else:
        #         print_row_fct = print_2d_row
        #     for row in arr:
        #         str_row = print_row_fct(row)
        #         print(str_row)
        #         if self.__output_file is not None:
        #             with open(path_arr, "a") as out_f:
        #                 out_f.write(str_row + "\n")


def get_random():
    val = str(random.randint(1, 10000000000))[1:8]
    # print(val)
    return val


class ParameterManager(dict):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--nb-cluster"] = int(self["--nb-cluster"])
        self["--nb-iteration"] = int(self["--nb-iteration"])

        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None
        self["--nb-iteration-palm"] = int(self["--nb-iteration-palm"]) if self["--nb-iteration-palm"] is not None else None

        self["--batch-assignation-time"] = int(self["--batch-assignation-time"]) if self["--batch-assignation-time"] is not None else None
        self["--assignation-time"] = int(self["--assignation-time"]) if self["--assignation-time"] is not None else None
        self["--nystrom"] = int(self["--nystrom"]) if self["--nystrom"] is not None else None

        self["--delta-threshold"] = float(self["--delta-threshold"])

        self["--minibatch"] = int(self["--minibatch"]) if self["--minibatch"] is not None else None
        self["--max-eval-train-size"] = int(self["--max-eval-train-size"]) if self["--max-eval-train-size"] is not None else None

        self.__init_nb_factors()
        self.__init_output_file()
        self.__init_seed()
        self.__init_dataset()

    def __init_dataset(self):
        if self["--blobs"] is not None:
            size_dim_clust = self["--blobs"].split("-")
            try:
                size_dim_clust = [int(elm) for elm in size_dim_clust]
                if len(size_dim_clust) != 3:
                    raise ValueError
            except ValueError:
                raise ValueError("Blobs chain malformed: {}. should be like 'size-dim-clusters'".format(self["--blobs"]))

            self["blobs_size"] = size_dim_clust[0]
            self["blobs_dim"] = size_dim_clust[1]
            self["blobs_clusters"] = size_dim_clust[2]

    def __init_nb_factors(self):
        if self["--nb-factors"] is not None:
            self["--nb-factors"] = int(self["--nb-factors"])

    def __init_output_file(self):
        out_file = get_random()
        self["--output-file"] = out_file
        if out_file is not None and len(out_file.split(".")) > 1:
            raise ValueError("Output file name should be given without any extension (no `.` in the string)")
        if out_file is not None:
            self["--output-file_resprinter"] = Path(out_file + "_results.csv")
            self["--output-file_objprinter"] = Path(out_file + "_objective.npz")
            self["--output-file_centroidprinter"] = Path(out_file + "_centroids.npy")

    def __init_seed(self):
        self["--seed"] = int(self["--seed"])
        if self["--seed"] is not None:
            np.random.seed(self["--seed"])
        else:
            self["--seed"] = int(self["--seed"])

    def get_dataset(self):
        """
        Return dataset in shape n x d.

        n: number of observations.
        d: dimensionality of observations.

        :return:
        """
        # todo normalize data before
        if self["--blobs"] is not None:
            blob_size = self["blobs_size"]
            blob_features = self["blobs_dim"]
            blob_centers = self["blobs_clusters"]
            return blobs_dataset(blob_size, blob_features, blob_centers)
        elif self["--caltech256"] is not None:
            caltech_size = int(self["--caltech256"])
            return caltech_dataset(caltech_size)
        elif self["--census"]:
            return census_dataset()
        elif self["--kddcup04"]:
            return kddcup04_dataset()
        elif self["--plants"]:
            return plants_dataset()
        elif self["--mnist"]:
            return mnist_dataset()
        elif self["--fashion-mnist"]:
            return fashion_mnist_dataset()
        elif self["--light-blobs"]:
            blob_size = 5000
            blob_features = 784
            blob_centers = 50
            return blobs_dataset(blob_size, blob_features, blob_centers)
        elif self["--lfw"]:
            return lfw_dataset(self["--seed"])
        elif self["--million-blobs"] is not None:
            return million_blobs_dataset(int(self["--million-blobs"]))
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

def caltech_dataset(caltech_size):
    data_dir = project_dir / "data/external" / "caltech256_{}.npz".format(caltech_size)
    loaded_npz = np.load(data_dir)
    return {
        "x_train": loaded_npz["x_train"],
        "y_train": loaded_npz["y_train"],
        "x_test": loaded_npz["x_test"],
        "y_test": loaded_npz["y_test"],
    }

def blobs_dataset(blob_size, blob_features, blob_centers):
    X, y = datasets.make_blobs(n_samples=blob_size, n_features=blob_features, centers=blob_centers, cluster_std=12)
    test_size = 1000
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    return {
        "x_train": X_train.reshape(X_train.shape[0], -1),
        "y_train": y_train,
        "x_test": X_test.reshape(X_test.shape[0], -1),
        "y_test": y_test
    }


def census_dataset():
    data_dir_obs = project_dir / "data/external" / "census.dat"
    X = np.memmap(data_dir_obs, mode="r", dtype="float32", shape=(2458285, 69))
    return {
        "x_train": X
    }


def kddcup04_dataset():
    data_dir_obs = project_dir / "data/external" / "kddcup04.dat"
    data_dir_labels = project_dir / "data/external" / "kddcup04.lab"
    X = np.memmap(data_dir_obs, mode="r", dtype="float32", shape=(145751, 74))
    y = np.memmap(data_dir_labels, mode="r", shape=(145751,))
    test_size = 5000
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
        "y_test": y_test
    }

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


def lfw_dataset(seed=None):
    lfw_data = fetch_lfw_people(min_faces_per_person=1, resize=0.4)
    X_train, X_test, y_train, y_test = train_test_split(lfw_data.data, lfw_data.target, test_size=0.33, random_state=seed)

    return {
        "x_train": X_train.reshape(X_train.shape[0], -1),
        "y_train": y_train,
        "x_test": X_test.reshape(X_test.shape[0], -1),
        "y_test": y_test
    }

def million_blobs_dataset(nb_million):
    data_dir_obs = project_dir / "data/external" / "blobs_{}_million.dat".format(nb_million)
    data_dir_labels = project_dir / "data/external" / "blobs_{}_million.lab".format(nb_million)
    X = np.memmap(data_dir_obs, mode="r", dtype="float32", shape=(int(1e6) * nb_million, 2000))
    y = np.memmap(data_dir_labels, mode="r", shape=(int(1e6) * nb_million,))
    test_size = 1000
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
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


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, examples, labels=None, batch_size=32, shuffle=True, to_categorical=False, return_indexes=False):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.examples = examples
        self.dim = examples.shape[1:]
        if self.labels is not None:
            self.n_classes = len(set(labels))
        self.shuffle = shuffle
        self.to_categorical = to_categorical
        if (self.to_categorical and not self.labels):
            raise AssertionError("Can't use 'to_categorical' if no labels are provided")
        self.return_indexes = return_indexes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.examples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        if self.return_indexes:
            return indexes
        else:
            # Generate data
            return self.__data_generation(indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.examples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim)).squeeze()
        y = np.empty((self.batch_size), dtype=int)


        # Generate data
        for i, idx in enumerate(indexes):
            # Store sample
            X[i,] = self.examples[idx]
            if self.labels is not None:
                # Store class
                y[i] = self.labels[idx]


        if self.to_categorical:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        elif self.labels is not None:
            return X, y
        else:
            return X

if __name__ == "__main__":
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    for d in DataGenerator(x):
        print(d)
    # for d in DataGenerator()


def visual_evaluation_palm4msa(target, init_factors, final_factors, result):
    nb_factors = len(init_factors)
    f = plt.figure(figsize=(10, 10))
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
        if i == 0:
            plt.ylabel("Final\nfactors", size='large')
        plt.title("$\mathbf{S}_" + str(nb_factors-i) + "^{final}$")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(final_factors[i].todense() if isinstance(final_factors[i], scipy.sparse.csr.csr_matrix) else final_factors[i])
        plt.subplot(3, nb_factors, nb_factors + nb_factors + (i+1))
        if i == 0:
            plt.ylabel("Init\nfactors", size='large')
        plt.xticks([])
        plt.yticks([])
        plt.title("$\mathbf{S}_" + str(nb_factors - i) + "^{init}$")
        plt.imshow(init_factors[i])

    plt.show()