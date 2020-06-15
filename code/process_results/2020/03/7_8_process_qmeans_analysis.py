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

def get_df_function(path):
    input_dir = results_dir / path
    df_results = get_df(input_dir)
    df_results = df_results[np.logical_not(df_results["failure"])]
    df_results = df_results.assign(results_dir=input_dir)
    return df_results

if __name__ == "__main__":
    results_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/results/")
    # suf_path = "2019/10/5_6_new_expes"
    suf_path = "2020/06/7_8_qmeans_more_iter"
    input_dir = results_dir / suf_path

    output_dir = "/home/luc/PycharmProjects/qalm_qmeans/results/processed/"
    output_dir = pathlib.Path(output_dir) / suf_path
    output_dir.mkdir(parents=True, exist_ok=True)

    not_processed_csv = output_dir / "notprocessed.csv"
    processed_csv = output_dir / "processed.csv"

    lst_input_path = [
        "2020/06/7_8_qmeans_only_big_largest_cluster_size",
        "2020/06/7_8_qmeans_more_iter_only_covtype",
        # "2020/06/7_8_qmeans_more_iter_only_covtype_no_npy",
        # "2020/06/7_8_qmeans_more_iter_other_seeds_no_npy",
    ]

    if True or not not_processed_csv.exists():
        # df_results = get_df(input_dir)
        # df_results = df_results[np.logical_not(df_results["failure"])]

        # df_results = pd.concat(map(get_df_function, lst_input_path))
        df_results = pd.concat([elm for elm in map(get_df_function, lst_input_path)])

        df_results.to_csv(not_processed_csv)
    else:
        df_results = pd.read_csv(not_processed_csv)

    dct_results = defaultdict(list)
    df_results = df_results[df_results["failure"] == False]

    # col_check_duplicates = [col for col in df_results.columns if (col.startswith("--") and "output-file" not in col)]
    # col_check_duplicates += ["kmeans", "qmeans", "palm"]

    # df_results.drop_duplicates(subset=col_check_duplicates, inplace=True)
    print(len(df_results))


    for i, row in df_results.iterrows():
        oar_id = int(row["oar_id"].split(".")[-1])
        # oar_id = int(row["oar_id"].split("_")[0])
        # if oar_id < 2008758:
        #     print("FOUND IT: {}. Skip".format(oar_id))
        #     continue
        if row["--breast-cancer"]:
            dct_results["dataset"].append("Breast Cancer")
        elif row["--caltech256"] != 'None':
            dct_results["dataset"].append("Caltech256 {}".format(row["--caltech256"]))
        elif row["--census"]:
            dct_results["dataset"].append("Census")
        elif row["--coil20"] != 'None':
            dct_results["dataset"].append("Coil20 {}".format(row["--coil20"]))
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
        else:
            raise ValueError("Unknown dataset")

        if row["kmeans"] and row["palm"]:
            dct_results["model"].append("Kmeans + Palm")
        elif row["kmeans"]:
            dct_results["model"].append("Kmeans")
        elif row["qmeans"]:
            dct_results["model"].append("QKmeans")
        else:
            raise ValueError("Model not Known")

        dct_results["results_dir"].append(row["results_dir"])

        dct_results["delta-treshold"].append(float(row["--delta-threshold"]))
        dct_results["batch-assignation-time-nb-sample"].append(int(row["--batch-assignation-time"]) if row["--batch-assignation-time"] is not None else np.nan)
        dct_results["ami-nb-sample"].append(int(row["--ami"]) if row["--ami"] is not None else np.nan)
        dct_results["nystrom-nb-sample"].append(int(row["--nystrom"]) if row["--nystrom"] is not None else np.nan)
        dct_results["1nn"].append(bool(row["--1-nn"]))
        dct_results["hierarchical-inside"].append(bool(row["--hierarchical"]))
        dct_results["hierarchical-init"].append(bool(row["--hierarchical-init"]))
        dct_results["initialization"].append(str(row["--initialization"]))
        dct_results["max-eval-train-size"].append(int(row["--max-eval-train-size"]) if row["--max-eval-train-size"] != 'None' else np.nan)
        dct_results["minibatch"].append(int(row["--minibatch"]) if row["--minibatch"] != 'None' else np.nan)
        dct_results["nb-cluster"].append(int(row["--nb-cluster"]))
        dct_results["nb-factors"].append(int(row["--nb-factors"]) if row["--nb-factors"] != 'None' else np.nan)
        dct_results["nb-iteration"].append(int(row["--nb-iteration"]))
        dct_results["nb-iteration-palm"].append(int(row["--nb-iteration-palm"]))
        dct_results["seed"].append(int(row["--seed"]) if row["--seed"] is not None else np.nan)
        dct_results["sparsity-factor"].append(int(row["--sparsity-factor"]) if row["--sparsity-factor"] != 'None' else np.nan)
        
        dct_results["path-centroids"].append(str((pathlib.Path(row["results_dir"]) / row["--output-file_centroidprinter"]).absolute()))
        dct_results["path-objective"].append(str((pathlib.Path(row["results_dir"]) / row["--output-file_objprinter"]).absolute()))
        
        dct_results["1nn-ball-tree-accuracy"].append(float(row["1nn_ball_tree_accuracy"]) if row["1nn_ball_tree_accuracy"] != 'None' else np.nan)
        dct_results["1nn-ball-tree-inference-time"].append(float(row["1nn_ball_tree_inference_time"]) if row["1nn_ball_tree_inference_time"] != 'None' else np.nan)
        
        dct_results["1nn-brute-accuracy"].append(float(row["1nn_brute_accuracy"]) if row["1nn_brute_accuracy"] != 'None' else np.nan)
        dct_results["1nn-brute-inference-time"].append(float(row["1nn_brute_inference_time"]) if row["1nn_brute_inference_time"] != 'None' else np.nan)
        
        dct_results["1nn-get-distance-time"].append(float(row["1nn_get_distance_time"]) if row["1nn_get_distance_time"] != 'None' else np.nan)
        
        dct_results["1nn-kd-tree-accuracy"].append(float(row["1nn_kd_tree_accuracy"]) if row["1nn_kd_tree_accuracy"] != 'None' else np.nan)
        dct_results["1nn-kd-tree-inference-time"].append(float(row["1nn_kd_tree_inference_time"]) if row["1nn_kd_tree_inference_time"] != 'None' else np.nan)
        
        dct_results["1nn-kmean-accuracy"].append(float(row["1nn_kmean_accuracy"]) if row["1nn_kmean_accuracy"] != 'None' else np.nan)
        dct_results["1nn-kmean-inference-time"].append(float(row["1nn_kmean_inference_time"]) if row["1nn_kmean_inference_time"] != 'None' else np.nan)
        
        dct_results["assignation-mean-time"].append(float(row["assignation_mean_time"]) if row["assignation_mean_time"] != 'None' else np.nan)
        dct_results["assignation-std-time"].append(float(row["assignation_std_time"]) if row["assignation_std_time"] != 'None' else np.nan)
        
        dct_results["batch-assignation-mean-time"].append(float(row["batch_assignation_mean_time"]) if row["batch_assignation_mean_time"] != 'None' else np.nan)
        dct_results["nb-param"].append(float(row["nb_param_centroids"]) if row["nb_param_centroids"] != 'None' else np.nan)
        dct_results["nb-flop"].append(dct_results["nb-param"][-1] * 2)
        dct_results["nystrom-build-time"].append(float(row["nystrom_build_time"]) if row["nystrom_build_time"] != 'None' else np.nan)
        
        
        dct_results["nystrom-inference-time"].append(float(row["nystrom_inference_time"]) if row["nystrom_inference_time"] != 'None' else np.nan)
        dct_results["nystrom-sampled-error-reconstruction"].append(float(row["nystrom_sampled_error_reconstruction"]) if row["nystrom_sampled_error_reconstruction"] != 'None' else np.nan)
        dct_results["nystrom-sampled-error-reconstruction-uniform"].append(float(row["nystrom_sampled_error_reconstruction_uniform"]) if row["nystrom_sampled_error_reconstruction_uniform"] != 'None' else np.nan)
        dct_results["nystrom-svm-accuracy"].append(float(row["nystrom_svm_accuracy"]) if row["nystrom_svm_accuracy"] != 'None' else np.nan)
        dct_results["nystrom-svm-time"].append(float(row["nystrom_svm_time"]) if row["nystrom_svm_time"] != 'None' else np.nan)
        dct_results["size-test"].append(float(row["size_test"]) if row["size_test"] != 'None' else np.nan)
        dct_results["size-train"].append(float(row["size_train"]) if row["size_train"] != 'None' else np.nan)
        dct_results["test-ami"].append(float(row["test_ami"]) if row["test_ami"] != 'None' else np.nan)
        dct_results["train-ami"].append(float(row["train_ami"]) if row["train_ami"] != 'None' else np.nan)
        dct_results["traintime"].append(float(row["traintime"]) if row["traintime"] is not None else np.nan)

        dct_results["dataset-dim"].append(int(dataset_dim[dct_results["dataset"][-1]]))

        nb_cluster = int(dct_results["nb-cluster"][-1])
        data_dim = int(dct_results["dataset-dim"][-1])
        nb_param_dense = nb_cluster * data_dim
        nb_param_compressed = dct_results["nb-param"][-1]
        dct_results["compression-rate"].append(nb_param_dense / nb_param_compressed)

        dct_results["final-objective-value"].append(float(row["final_objective_value"]))



    final_df = DataFrame.from_dict(dct_results)
    print(len(final_df))
    final_df.to_csv(processed_csv)