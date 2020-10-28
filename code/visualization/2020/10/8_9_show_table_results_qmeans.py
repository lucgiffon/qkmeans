import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
from collections import defaultdict
import re
from io import StringIO
from pandas.errors import EmptyDataError
from qkmeans.utils import logger
from visualization.utils import get_dct_result_files_by_root, build_df
from textwrap import wrap
import matplotlib
import logging
import plotly.io as pio

import plotly.graph_objects as go

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)
font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16
        }
matplotlib.rc('font', **font)

template_error_bars = go.layout.Template()
go_scatter_error_bar = go.Scatter(
    error_y=dict(
        type='data',
        # color='black',
        thickness=0.8,
        width=3,
    ))
template_error_bars.data.scatter = [go_scatter_error_bar]
pio.templates["template_error_bars"] = template_error_bars
pio.templates.default = "plotly_white+template_error_bars"

def get_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir)
    col_to_delete = []

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return df_results

def get_df_from_path(path):
    input_path = results_dir / path / "processed.csv"
    df_results = pd.read_csv(input_path)
    df_results = df_results.fillna("None")
    return df_results

if __name__ == "__main__":
    results_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/results/processed")

    # suf_path = "2020/06/7_8_qmeans_more_iter"
    suf_path = "2020/10/8_9_qmeans_more_iter"
    suf_path_small = "2020/06/7_8_qmeans_more_iter_small_datasets"
    df_results_big = get_df_from_path(suf_path)
    df_results_small = get_df_from_path(suf_path_small)
    df_results_small = df_results_small.loc[~df_results_small["dataset"].isin(["Caltech256 32", "Kddcup99", "Coverage Type"])]
    df_results = pd.concat([df_results_small, df_results_big])

    suf_path_efficient = "2020/01/0_0_efficient_nystrom_bis_bis"
    input_dir_efficient = results_dir / suf_path_efficient
    processed_csv_efficient = input_dir_efficient / "processed.csv"
    df_results_efficient = pd.read_csv(processed_csv_efficient)
    df_results_efficient = df_results_efficient.fillna("None")

    figures_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/reports/figures/")
    output_dir = figures_dir / suf_path / "histogrammes"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        "nb-flop",
        "nb-param",
        "compression-rate",
        "1nn-kmean-accuracy",
        "nystrom-sampled-error-reconstruction",
        "nystrom-svm-accuracy",
        "test-ami",
        # "train-ami",

    ]

    other_nystrom_efficient_methods = [
        "nystrom-sampled-error-reconstruction-uniform",
        "nystrom-sampled-error-reconstruction-uop",
        "nystrom-sampled-error-reconstruction-uop-kmeans",
    ]
    other_nystrom_efficient_methods_accuracy = [
        "nystrom-svm-accuracy-uniform",
        "nystrom-svm-accuracy-uop",
        "nystrom-svm-accuracy-uop-kmeans",
    ]
    other_1nn_methods_accuracy = [
        "1nn-kd-tree-accuracy",
        "1nn-brute-accuracy",
        "1nn-ball-tree-accuracy"
    ]

    y_axis_scale_by_task = {
        "assignation-mean-time": "linear",
        "test-ami": "linear",
        "train-ami": "linear",
        "1nn-kmean-inference-time": "linear",
        "nystrom-build-time": "linear",
        "nystrom-inference-time": "linear",
        "nystrom-sampled-error-reconstruction": "log",
        "traintime": "linear",
        "1nn-kmean-accuracy": "linear",
        "batch-assignation-mean-time": "linear",
        "1nn-get-distance-time": "linear",
        "nystrom-svm-accuracy": "linear",
        "nystrom-svm-time": "linear",
        "nb-param": "log",
        "nb-flop-centroids": "log"

    }

    y_axis_label_by_task = {
        "assignation-mean-time": "time (s)",
        "test-ami": "ami",
        "train-ami": "ami",
        "1nn-kmean-inference-time": "time (s)",
        "nystrom-build-time": "time (s)",
        "nystrom-inference-time": "time (s)",
        "nystrom-sampled-error-reconstruction": "log(error)",
        "traintime": "time (s)",
        "1nn-kmean-accuracy": "accuracy",
        "batch-assignation-mean-time": "time (s)",
        "1nn-get-distance-time": "time(s)",
        "nystrom-svm-accuracy": "accuracy",
        "nystrom-svm-time": "time(s)",
        "nb-param": "log(# non-zero values)",
        "nb-flop-centroids": "log(# FLOP)"
    }

    color_by_sparsity = {
        2: (0, 128, 0),  # green
        3: (0, 0, 255),  # blue
        5: (0, 153, 153)  # turquoise
    }

    dct_name_legend = {
        "nystrom-sampled-error-reconstruction-uniform": "Uniform sampling",
        "nystrom-sampled-error-reconstruction-uop": "Fast-Nys",
        "nystrom-sampled-error-reconstruction-uop-kmeans": "K Fast-Kys",
        "nystrom-svm-accuracy-uniform": "Uniform sampling",
        "nystrom-svm-accuracy-uop": "Fast-Nys",
        "nystrom-svm-accuracy-uop-kmeans": "K Fast-Kys",
    }

    color_by_init = {
        "kmeans++": (0, 128, 0),  # green
        "uniform_sampling": (255, 0, 0)  # ref
    }

    dct_legend_color = {
        "QK-means sparsity 2": "dodgerblue",
        "QK-means sparsity 3": "blue",
        "QK-means sparsity 5": "purple",
        "Uniform sampling": "gray",
        "Fast-Nys": "orange",
        "K Fast-Kys": "olive"

    }

    datasets = set(df_results["dataset"].values)


    nb_iter_palm = set(df_results["nb-iteration-palm"].values)
    # hierarchical_values = set(df_results["hierarchical-init"])


    df_histo = df_results[df_results["nb-iteration-palm"] == max(nb_iter_palm)]
    df_histo = df_histo[df_histo["initialization"] == "kmeans++"]
    df_histo = df_histo.loc[(df_histo["sparsity-factor"] == "None") | (df_histo["sparsity-factor"] == 3)]
    df_histo = df_histo.loc[~(df_histo["dataset"]== "Coverage Type") | ((df_histo["dataset"]== "Coverage Type") & (df_histo["nb-cluster"] == 256))]

    # df_histo = df_histo[df_histo["dataset"]== "Coverage Type"]

    dct_data_correct_lambda = {
        "Coil20 32": 10,
        "Caltech256 32": 1000,
        "Mnist": 3000,
        "Coverage Type": 1000,
        "Fashion Mnist": 2000,
        "Kddcup99": 5,
        "Breast Cancer": 700,
    }

    i=0
    for data in datasets:
        # if i == 2: exit()
        i += 1


        df_data = df_histo[df_histo["dataset"] == data]
        df_data_efficient = df_results_efficient[df_results_efficient["dataset"] == data]
        nb_clusters = sorted(set(df_data["nb-cluster"].values))
        print("##################################################################################################")
        print(data, "nb clusters", nb_clusters)
        for task in tasks:
            print("----------------------------------------------------------------------------------------------")
            print(task, data, nb_clusters)

            init_schemes = set(df_data["initialization"].values)
            for init in init_schemes:
                title_fig_sparsity_figures = "{} {} {}"

                df_init = df_data[df_data["initialization"] == init]

                ###########
                # KMEANS #
                ###########
                df_kmeans = df_init[df_init["model"] == "Kmeans"]


                #################
                # KMEANS VANILLA#
                #################

                df_kmeans_vanilla = df_kmeans[~(df_kmeans["L1-proj"] == True)]

                task_values_mean = [df_kmeans_vanilla[df_kmeans_vanilla["nb-cluster"] == nb_cluster][task].mean() for nb_cluster in nb_clusters]
                task_values_std = [df_kmeans_vanilla[df_kmeans_vanilla["nb-cluster"] == nb_cluster][task].std() for nb_cluster in nb_clusters]

                print("Kmeans", task_values_mean)

                ###########
                # QKMEANS #
                ###########
                df_qkmeans = df_init[df_init["model"] == "QKmeans"]

                for hierarchical_value in [True]:
                    df_hierarchical = df_qkmeans[df_qkmeans["hierarchical-init"] == hierarchical_value]
                    sparsity_values = set(df_hierarchical["sparsity-factor"].values)
                    # sparsity_values.remove("None")
                    for sparsity_value in sparsity_values:
                        df_sparsity = df_hierarchical[df_hierarchical["sparsity-factor"] == sparsity_value]
                        task_values_mean = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].mean() for nb_cluster in nb_clusters]
                        task_values_std = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].std() for nb_cluster in nb_clusters]

                        title_figure = title_fig_sparsity_figures.format(data, task, init)

                        trace_name = "QK-means sparsity {}".format(int(sparsity_value))
                        print(trace_name, task_values_mean)
                        if task == "compression-rate":
                            qkmeans_compression_rate = task_values_mean[0]


                #################
                # KMEANS L1 PROJ#
                #################

                df_kmeans_l1_proj = df_kmeans[df_kmeans["L1-proj"] == True]
                set_lambda_values = sorted(set(df_kmeans_l1_proj["--lambda-l1-proj"].values))
                max_compression_rate = 1
                # for lambda_val in set_lambda_values:
                lambda_val = dct_data_correct_lambda[data]
                df_lambda = df_kmeans_l1_proj[df_kmeans_l1_proj["--lambda-l1-proj"] == lambda_val]
                task_values_mean = [df_lambda[df_lambda["nb-cluster"] == nb_cluster][task].mean() for nb_cluster in nb_clusters]
                task_values_std = [df_lambda[df_lambda["nb-cluster"] == nb_cluster][task].std() for nb_cluster in nb_clusters]
                trace_name = "Kmeans L1 proj {}".format(int(lambda_val))
                print(trace_name, task_values_mean)
                # if task == "compression-rate":
                #     max_compression_rate = max(max_compression_rate, task_values_mean[0])
                #
                # if task == "compression-rate" and max_compression_rate == 1:
                #     print(data, data,data, data,data, data,data, data,data, data,data, data,data, data)

                #####################
                # Efficient NYSTROM #
                #####################
                if "nystrom-sampled-error-reconstruction" == task:
                    for idx_other_nystrom, str_other_nystrom in enumerate(other_nystrom_efficient_methods):
                        task_values_mean = [df_data_efficient[df_data_efficient["nb-landmarks"] == nb_cluster][str_other_nystrom].mean() for nb_cluster in nb_clusters]
                        task_values_std = [df_data_efficient[df_data_efficient["nb-landmarks"] == nb_cluster][str_other_nystrom].std() for nb_cluster in nb_clusters]

                        print(str_other_nystrom, task_values_mean)

                if "nystrom-svm-accuracy" == task:
                    for idx_other_nystrom, str_other_nystrom in enumerate(other_nystrom_efficient_methods_accuracy):
                        task_values_mean = [df_data_efficient[df_data_efficient["nb-landmarks"] == nb_cluster][str_other_nystrom].mean() for nb_cluster in nb_clusters]
                        task_values_std = [df_data_efficient[df_data_efficient["nb-landmarks"] == nb_cluster][str_other_nystrom].std() for nb_cluster in nb_clusters]

                        print(str_other_nystrom, task_values_mean)

                #######
                # 1NN #
                #######
                if "1nn" in task:
                    for idx_other_1nn, std_other_1nn in enumerate(other_1nn_methods_accuracy):
                        try:
                            task_values_mean = [df_kmeans[df_kmeans["nb-cluster"] == nb_cluster][std_other_1nn].mean() for nb_cluster in nb_clusters ]
                            task_values_std = [df_kmeans[df_kmeans["nb-cluster"] == nb_cluster][std_other_1nn].std() for nb_cluster in nb_clusters]
                        except:
                            task_values_mean = [np.nan for nb_cluster in nb_clusters]
                            task_values_std = [np.nan for nb_cluster in nb_clusters]

                        print(std_other_1nn, task_values_mean)

