import logging
from copy import deepcopy

from pathlib import Path

from sklearn import datasets

from numpy import eye
import numpy as np
from numpy.linalg import multi_dot
import daiquiri
from pyqalm.projection_operators import prox_splincol, prox_splincol_fast

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


def get_lambda_proxsplincol_fast(nb_keep_values):
    return lambda mat: prox_splincol_fast(mat, nb_keep_values)


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
        self.__dicts = []
        self.__dicts.extend(args)
        self.__header = header
        self.__output_file = output_file

    def _get_ordered_items(self):
        all_keys = []
        all_values = []
        for d in self.__dicts:
            keys, values = zip(*d.items())
            all_keys.extend(keys)
            all_values.extend(values)
        arr_keys, arr_values = np.array(all_keys), np.array(all_values)
        indexes_sort = np.argsort(arr_keys)
        return list(arr_keys[indexes_sort]), list(arr_values[indexes_sort])

    def _get_values_ordered_by_keys(self):
        _, values = self._get_ordered_items()
        return values

    def _get_ordered_keys(self):
        keys, _ = self._get_ordered_items()
        return keys

    def add(self, d):
        """
        Add dictionnary after initialisation.

        :param d: the dictionnary object you want to add.
        :return:
        """
        self.__dicts.append(d)

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

        assert len(cols) == len(arr[0]) or (len(arr.shape) == 1 and len(cols) == 1)

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

class ParameterManager(dict):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--nb-cluster"] = int(self["--nb-cluster"])
        self["--nb-iteration"] = int(self["--nb-iteration"])

        self.__init_output_file()
        self.__init_seed()

    def __init_output_file(self):
        out_file = self["--output-file"]
        if out_file is not None and len(out_file.split(".")) > 1:
            raise ValueError("Output file name should be given without any extension (no `.` in the string)")
        if out_file is not None:
            self["--output-file_resprinter"] = Path(out_file + "_results.csv")
            self["--output-file_objprinter"] = Path(out_file + "_objective.csv")

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
        if self["--blobs"]:
            return blobs_dataset()
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


class ParameterManagerQmeans(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--sparsity-factor"] = int(self["--sparsity-factor"])
        self["--nb-factors"] = int(self["--nb-factors"])
        self["--nb-iteration-palm"] = int(self["--nb-iteration-palm"])

def blobs_dataset():
    X, _ = datasets.make_blobs(n_samples=1000, n_features=20, centers=50)
    return X
