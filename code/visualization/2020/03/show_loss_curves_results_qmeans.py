import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
from io import StringIO
from pandas.errors import EmptyDataError
from qkmeans.utils import logger
from visualization.utils import get_dct_result_files_by_root, build_df
from textwrap import wrap
import matplotlib
import logging

import plotly.graph_objects as go

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)
font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16
        }

matplotlib.rc('font', **font)

def get_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir)
    col_to_delete = []

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return df_results


if __name__ == "__main__":
    results_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/results/processed")

    suf_path = "2020/03/6_7_qmeans_all_only_small/"
    input_dir = results_dir / suf_path
    processed_csv = input_dir / "processed.csv"
    df_results = pd.read_csv(processed_csv)
    df_results = df_results.fillna("None")

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
        # "assignation-mean-time",
        # "1nn-kmean-inference-time",
        # "nystrom-build-time",
        # "nystrom-inference-time",
        "nystrom-sampled-error-reconstruction",
        "test-ami",
        "train-ami",
        # "traintime",
        # "1nn-kmean-accuracy",
        # "batch-assignation-mean-time",
        # "1nn-get-distance-time",
        # "nystrom-svm-accuracy",
        # "nystrom-svm-time",
        # "nb-param-centroids",
        # "nb-flop-centroids"
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
        "nb-param-centroids": "log",
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
        "nb-param-centroids": "log(# non-zero values)",
        "nb-flop-centroids": "log(# FLOP)"
    }

    datasets = set(df_results["dataset"].values)
    sparsity_values = set(df_results["sparsity-factor"].values)
    sparsity_values.remove("None")
    nb_clusters = sorted(set(df_results["nb-cluster"].values))
    nb_iter_palm = set(df_results["nb-iteration-palm"].values)
    # hierarchical_values = set(df_results["hierarchical-init"])

    df_histo = df_results[df_results["nb-iteration-palm"] == max(nb_iter_palm)]
    for data in datasets:
        df_data = df_histo[df_histo["dataset"] == data]
        for task in tasks:
            fig = go.Figure()

            ###########
            # QKMEANS #
            ###########
            df_qkmeans = df_data[df_data["model"] == "QKmeans"]

            for hierarchical_value in [True]:
                df_hierarchical = df_qkmeans[df_qkmeans["hierarchical-init"] == hierarchical_value]
                for sparsity_value in sparsity_values:
                    df_sparsity = df_hierarchical[df_hierarchical["sparsity-factor"] == sparsity_value]

                    task_values_mean = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].mean() for nb_cluster in nb_clusters]
                    task_values_std = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].std() for nb_cluster in nb_clusters]
                    fig.add_trace(go.Bar(
                        x=nb_clusters,
                        y=task_values_mean,
                        error_y=dict(
                            type='data',  # value of error bar given in data coordinates
                            array=task_values_std,
                            visible=True
                        ),
                        name='Sparsity Value {}'.format(sparsity_value),
                        # marker_color='indianred'
                    ))

            ###########
            # KMEANS #
            ###########
            df_kmeans = df_data[df_data["model"] == "Kmeans"]

            task_values_mean = [df_kmeans[df_kmeans["nb-cluster"] == nb_cluster][task].mean() for nb_cluster in nb_clusters]
            task_values_std = [df_kmeans[df_kmeans["nb-cluster"] == nb_cluster][task].std() for nb_cluster in nb_clusters]

            fig.add_trace(go.Bar(
                x=nb_clusters,
                y=task_values_mean,
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=task_values_std,
                    visible=True
                ),
                name='Kmeans',
                # marker_color='indianred'
            ))


            ##################
            # KMEANS  + PALM #
            ##################
            df_kmeans_palm = df_data[df_data["model"] == "Kmeans + Palm"]

            for hierarchical_value in [True]:
                df_hierarchical = df_kmeans_palm[df_kmeans_palm["hierarchical-inside"] == hierarchical_value]
                for sparsity_value in sparsity_values:
                    df_sparsity = df_hierarchical[df_hierarchical["sparsity-factor"] == sparsity_value]

                    task_values_mean = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].mean() for nb_cluster in nb_clusters]
                    task_values_std = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].std() for nb_cluster in nb_clusters]
                    fig.add_trace(go.Bar(
                        x=nb_clusters,
                        y=task_values_mean,
                        error_y=dict(
                            type='data',  # value of error bar given in data coordinates
                            array=task_values_std,
                            visible=True
                        ),
                        name='KP Sparsity Value {}'.format(sparsity_value),
                        # marker_color='indianred'
                    ))

            title = "{} {}".format(data, task)

            fig.update_layout(barmode='group',
                              title=task,
                              xaxis_title="# Cluster",
                              yaxis_title=y_axis_label_by_task[task],
                              yaxis_type=y_axis_scale_by_task[task],
                              xaxis={'type': 'category'},
                              )
            fig.show()


