from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
from io import StringIO

from pandas import DataFrame
from pandas.errors import EmptyDataError
from qkmeans.utils import logger
from visualization.utils import get_dct_result_files_by_root, build_df
from textwrap import wrap
import matplotlib
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)
font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16
        }

matplotlib.rc('font', **font)

def get_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir, tpl_results=("results","objective"))
    col_to_delete = []

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return df_results


dataset_dim = {
        "Fashion Mnist": 784,
        "Mnist": 784,
        "Caltech256 32": 32*32*3,
        "Blobs": 2000,
        "Kddcup04": 74,
        "Kddcup99": 116,
        "Census": 68,
        "Plants": 70,
        "Breast Cancer": 30,
        "Coverage Type": 54,
        "Coil20 32": 32*32
    }

if __name__ == "__main__":
    results_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/results/")
    # suf_path = "2019/10/5_6_new_expes"
    suf_path = "2020/01/0_0_efficient_nystrom_bis_bis"
    input_dir = results_dir / suf_path
    suf_path_coil = "2020/06/1_2_efficient_nystrom_coil20"
    input_dir_coil = results_dir / suf_path_coil

    output_dir = "/home/luc/PycharmProjects/qalm_qmeans/results/processed/"
    output_dir = pathlib.Path(output_dir) / suf_path
    output_dir.mkdir(parents=True, exist_ok=True)

    not_processed_csv = output_dir / "notprocessed.csv"
    processed_csv = output_dir / "processed.csv"

    # if not not_processed_csv.exists():
    df_results = get_df(input_dir)
    df_results_coil = get_df(input_dir_coil)
    df_results = pd.concat([df_results, df_results_coil])
    df_results = df_results[np.logical_not(df_results["failure"])]

    df_results.to_csv(not_processed_csv)
    # else:
    #     df_results = pd.read_csv(not_processed_csv)

    dct_results = defaultdict(list)

    for i, row in df_results.iterrows():
        if row["--breast-cancer"]:
            dct_results["dataset"].append("Breast Cancer")
        elif row["--caltech256"] != 'None':
            dct_results["dataset"].append("Caltech256 {}".format(row["--caltech256"]))
        elif row["--census"]:
            dct_results["dataset"].append("Census")
        # elif row["--coil20"] != 'None':
        #     dct_results["dataset"].append("Coil20 {}".format(row["--coil20"]))
        elif row["--covtype"]:
            dct_results["dataset"].append("Coverage Type")
        elif row["--fashion-mnist"]:
            dct_results["dataset"].append("Fashion Mnist")
        elif row["--mnist"]:
            dct_results["dataset"].append("Mnist")
        elif row["--kddcup04"]:
            dct_results["dataset"].append("Kddcup04")
        elif row["--kddcup99"]:
            dct_results["dataset"].append("Kddcup99")
        elif row["--lfw"]:
            dct_results["dataset"].append("LFW")
        elif row["--blobs"] != 'None':
            dct_results["dataset"].append("Blobs {}".format(row["--blobs"]))
        elif row["--light-blobs"]:
            dct_results["dataset"].append("Light blobs")
        elif row["--million-blobs"] != 'None':
            dct_results["dataset"].append("Million blobs {}".format(row["--milion-blobs"]))
        elif row["--plants"]:
            dct_results["dataset"].append("Plants")
        elif row["--coil20"] != 'None':
            dct_results["dataset"].append("Coil20 {}".format(int(row["--coil20"])))
        else:
            raise ValueError("Unknown dataset")

        dct_results["nystrom-nb-sample"].append(int(row["--nystrom"]) if row["--nystrom"] is not None else np.nan)
        # dct_results["initialization"].append(str(row["--initialization"]))
        dct_results["max-eval-train-size"].append(int(row["--max-eval-train-size"]) if row["--max-eval-train-size"] != 'None' else np.nan)
        dct_results["minibatch"].append(int(row["--minibatch"]) if row["--minibatch"] != 'None' else np.nan)
        dct_results["nb-iteration"].append(int(row["--nb-iteration"]))
        dct_results["nb-landmarks"].append(int(row["--nb-landmarks"]))
        dct_results["seed"].append(int(row["--seed"]) if row["--seed"] is not None else np.nan)

        dct_results["nystrom-sampled-error-reconstruction-kmeans"].append(float(row["nystrom_sampled_error_reconstruction_kmeans"]) if row["nystrom_sampled_error_reconstruction_kmeans"] != 'None' else np.nan)
        dct_results["nystrom-sampled-error-reconstruction-uop-kmeans"].append(float(row["nystrom_sampled_error_reconstruction_uop_kmeans"]) if row["nystrom_sampled_error_reconstruction_uop_kmeans"] != 'None' else np.nan)
        dct_results["nystrom-sampled-error-reconstruction-uop"].append(float(row["nystrom_sampled_error_reconstruction_uop"]) if row["nystrom_sampled_error_reconstruction_uop"] != 'None' else np.nan)
        dct_results["nystrom-sampled-error-reconstruction-uniform"].append(float(row["nystrom_sampled_error_reconstruction_uniform"]) if row["nystrom_sampled_error_reconstruction_uniform"] != 'None' else np.nan)
        dct_results["nystrom-sampled-error-reconstruction-seeds"].append(float(row["nystrom_sampled_error_reconstruction_seeds"]) if row["nystrom_sampled_error_reconstruction_seeds"] != 'None' else np.nan)

        dct_results["nystrom-svm-accuracy-kmeans"].append(float(row["nystrom_svm_accuracy_kmeans"]) if row["nystrom_svm_accuracy_kmeans"] != 'None' else np.nan)
        dct_results["nystrom-svm-accuracy-uop-kmeans"].append(float(row["nystrom_svm_accuracy_uop_kmeans"]) if row["nystrom_svm_accuracy_uop_kmeans"] != 'None' else np.nan)
        dct_results["nystrom-svm-accuracy-uop"].append(float(row["nystrom_svm_accuracy_uop"]) if row["nystrom_svm_accuracy_uop"] != 'None' else np.nan)
        dct_results["nystrom-svm-accuracy-uniform"].append(float(row["nystrom_svm_accuracy_uniform"]) if row["nystrom_svm_accuracy_uniform"] != 'None' else np.nan)
        dct_results["nystrom-svm-accuracy-seeds"].append(float(row["nystrom_svm_accuracy_seeds"]) if row["nystrom_svm_accuracy_seeds"] != 'None' else np.nan)

        dct_results["size-test"].append(float(row["size_test"]) if row["size_test"] != 'None' else np.nan)
        dct_results["size-train"].append(float(row["size_train"]) if row["size_train"] != 'None' else np.nan)

        dct_results["dataset-dim"].append(int(dataset_dim[dct_results["dataset"][-1]]))

    final_df = DataFrame.from_dict(dct_results)

    final_df.to_csv(processed_csv)