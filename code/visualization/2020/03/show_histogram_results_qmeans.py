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

    suf_path = "2020/06/7_8_qmeans_more_iter"
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
        "nystrom-sampled-error-reconstruction",
        # "test-ami",
        # "train-ami",
        # "1nn-kmean-accuracy",
        "nystrom-svm-accuracy",
        "nb-param",
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
    sparsity_values = set(df_results["sparsity-factor"].values)
    sparsity_values.remove("None")

    nb_iter_palm = set(df_results["nb-iteration-palm"].values)
    # hierarchical_values = set(df_results["hierarchical-init"])


    df_histo = df_results[df_results["nb-iteration-palm"] == max(nb_iter_palm)]
    df_histo = df_histo[df_histo["initialization"] == "kmeans++"]

    # df_histo = df_histo[df_histo["dataset"]== "Coverage Type"]

    i=0
    for data in datasets:
        # if i == 2: exit()
        i += 1

        df_data = df_histo[df_histo["dataset"] == data]
        df_data_efficient = df_results_efficient[df_results_efficient["dataset"] == data]
        nb_clusters = sorted(set(df_data["nb-cluster"].values))

        for task in tasks:

            init_schemes = set(df_data["initialization"].values)
            for init in init_schemes:
                dct_sparsity_figure = defaultdict(lambda: go.Figure())
                title_fig_sparsity_figures = "{} {} {}"

                df_init = df_data[df_data["initialization"] == init]

                ###########
                # QKMEANS #
                ###########
                df_qkmeans = df_init[df_init["model"] == "QKmeans"]

                for hierarchical_value in [True]:
                    df_hierarchical = df_qkmeans[df_qkmeans["hierarchical-init"] == hierarchical_value]
                    for sparsity_value in sparsity_values:
                        df_sparsity = df_hierarchical[df_hierarchical["sparsity-factor"] == sparsity_value]
                        task_values_mean = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].mean() for nb_cluster in nb_clusters]
                        task_values_std = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].std() for nb_cluster in nb_clusters]

                        title_figure = title_fig_sparsity_figures.format(data, task, init)

                        trace_name = "QK-means sparsity {}".format(int(sparsity_value))

                        dct_sparsity_figure[title_figure].add_trace(go.Bar(
                            x=nb_clusters,
                            y=task_values_mean,
                            error_y=dict(
                                type='data',  # value of error bar given in data coordinates
                                array=task_values_std,
                                visible=True
                            ),
                            marker_color=dct_legend_color[trace_name],
                            hovertext=trace_name,
                            name=trace_name,
                            # marker_color='indianred'
                        ))

                ###########
                # KMEANS #
                ###########
                df_kmeans = df_init[df_init["model"] == "Kmeans"]

                task_values_mean = [df_kmeans[df_kmeans["nb-cluster"] == nb_cluster][task].mean() for nb_cluster in nb_clusters]
                task_values_std = [df_kmeans[df_kmeans["nb-cluster"] == nb_cluster][task].std() for nb_cluster in nb_clusters]

                for title_figure, fig in dct_sparsity_figure.items():
                    fig.add_trace(go.Bar(
                        x=nb_clusters,
                        y=task_values_mean,
                        error_y=dict(
                            type='data', # value of error bar given in data coordinates
                            array=task_values_std,
                            visible=True
                        ),
                        marker_color='red',
                        name='Kmeans',
                        # marker_color='indianred'
                    ))


                ##################
                # KMEANS  + PALM #
                ##################
                # df_kmeans_palm = df_init[df_init["model"] == "Kmeans + Palm"]
                #
                # for hierarchical_value in [True]:
                #     df_hierarchical = df_kmeans_palm[df_kmeans_palm["hierarchical-inside"] == hierarchical_value]
                #     for sparsity_value in sparsity_values:
                #         df_sparsity = df_hierarchical[df_hierarchical["sparsity-factor"] == sparsity_value]
                #
                #         task_values_mean = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].mean() for nb_cluster in nb_clusters]
                #         task_values_std = [df_sparsity[df_sparsity["nb-cluster"] == nb_cluster][task].std() for nb_cluster in nb_clusters]
                #
                #         title_figure = title_fig_sparsity_figures.format(data, task, int(sparsity_value), init)
                #         dct_sparsity_figure[title_figure].add_trace(go.Bar(
                #             x=nb_clusters,
                #             y=task_values_mean,
                #             error_y=dict(
                #                 type='data',  # value of error bar given in data coordinates
                #                 array=task_values_std,
                #                 visible=True
                #             ),
                #             marker_color='blue',
                #             name='Kmeans + PALM',
                #             # marker_color='indianred'
                #         ))


                #####################
                # Efficient NYSTROM #
                #####################
                for title_figure, fig in dct_sparsity_figure.items():
                    if "nystrom-sampled-error-reconstruction" == task:
                        for idx_other_nystrom, str_other_nystrom in enumerate(other_nystrom_efficient_methods):
                            task_values_mean = [df_data_efficient[df_data_efficient["nb-landmarks"] == nb_cluster][str_other_nystrom].mean() for nb_cluster in nb_clusters]
                            task_values_std = [df_data_efficient[df_data_efficient["nb-landmarks"] == nb_cluster][str_other_nystrom].std() for nb_cluster in nb_clusters]
                            fig.add_trace(go.Bar(
                                x=nb_clusters,
                                y=task_values_mean,
                                error_y=dict(
                                    type='data',  # value of error bar given in data coordinates
                                    array=task_values_std,
                                    visible=True
                                ),
                                name=dct_name_legend[str_other_nystrom],
                                marker_color=dct_legend_color[dct_name_legend[str_other_nystrom]]
                            ))

                    if "nystrom-svm-accuracy" == task:
                        for idx_other_nystrom, str_other_nystrom in enumerate(other_nystrom_efficient_methods_accuracy):
                            task_values_mean = [df_data_efficient[df_data_efficient["nb-landmarks"] == nb_cluster][str_other_nystrom].mean() for nb_cluster in nb_clusters]
                            task_values_std = [df_data_efficient[df_data_efficient["nb-landmarks"] == nb_cluster][str_other_nystrom].std() for nb_cluster in nb_clusters]
                            fig.add_trace(go.Bar(
                                x=nb_clusters,
                                y=task_values_mean,
                                error_y=dict(
                                    type='data',  # value of error bar given in data coordinates
                                    array=task_values_std,
                                    visible=True
                                ),
                                name=dct_name_legend[str_other_nystrom],
                                marker_color=dct_legend_color[dct_name_legend[str_other_nystrom]]
                            ))



                for title_figure, fig in dct_sparsity_figure.items():
                    if "nystrom" in title_figure:
                        x_axis_title = "# Landmarks"
                    else:
                        x_axis_title = "# Cluster"

                    if task == "nb-param":
                        showlegend = True
                        x_legend = 0.01
                        y_legend = 1
                    else:
                        showlegend = False
                        x_legend = 0
                        y_legend = -0.3
                    fig.update_layout(barmode='group',
                                      width=1000,
                                      height=350,
                                      autosize=False,
                                      margin=dict(l=20, r=20, t=20, b=20),
                                      # title=title_figure,
                                      xaxis_title=x_axis_title,
                                      yaxis_title=y_axis_label_by_task[task],
                                      yaxis_type=y_axis_scale_by_task[task],
                                      xaxis={'type': 'category'},
                                      font=dict(
                                          # family="Courier New, monospace",
                                          size=18,
                                          color="black"
                                      ),
                                      legend_orientation="h",
                                      showlegend=showlegend,
                                      legend=dict(
                                          x=x_legend, y=y_legend,
                                          traceorder="normal",
                                          font=dict(
                                              family="sans-serif",
                                              size=18,
                                              color="black"
                                          ),
                                          # bgcolor="LightSteelBlue",
                                          # bordercolor="Black",
                                          borderwidth=1,
                                        )
                                      )
                    # fig.show()
                    output_dir_final = output_dir / data / task
                    output_dir_final.mkdir(parents=True, exist_ok=True)
                    fig.write_image(str((output_dir_final / title_figure.replace(" ", "_")).absolute()) + ".png")


