import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
from io import StringIO
from pandas.errors import EmptyDataError
from pyqalm.utils import logger
from visualization.utils import get_dct_result_files_by_root, build_df
from textwrap import wrap
import matplotlib


font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 12
        }

matplotlib.rc('font', **font)

def get_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir)
    col_to_delete = ["--initialization",
                     "--1-nn",
                     "--help",
                     "--kddcup",
                     "--census",
                     "--nb-factors",
                     "--nb-iteration",
                     "--nb-iteration-palm",
                     "--plants",
                     "--verbose",
                     "--output-file",
                     "--output-file_centroidprinter",
                     "--output-file_objprinter",
                     "--output-file_resprinter",]

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return df_results


if __name__ == "__main__":
    # suf_path = "2019/07/qmeans_analysis_blobs_log2_clusters_bis"
    suf_path = "2019/07/qmeans_analysis_blobs_log2_clusters_bis_higherdim"
    input_dir = "/home/luc/PycharmProjects/qalm_qmeans/results/" + suf_path
    output_dir = "/home/luc/PycharmProjects/qalm_qmeans/reports/figures/"+ "2019/07/tech_report" + "/histogrammes"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suf_path_1024 = "2019/07/qmeans_analysis_blobs_log2_clusters_bis_higherdim_1024"
    input_dir_1024 = "/home/luc/PycharmProjects/qalm_qmeans/results/" + suf_path_1024


    df_results_no_1024 = get_df(input_dir)
    # df_results_1024 = get_df(input_dir_1024)
    # df_results = pd.concat([df_results_no_1024, df_results_1024])
    df_results = df_results_no_1024
    df_results_kmeans = df_results[df_results["kmeans"]]

    df_results = pd.concat([df_results, df_results_kmeans])
    df_results_failure = df_results[df_results["failure"]]
    df_results = df_results[np.logical_not(df_results["failure"])]

    df_results_qmeans = df_results[df_results["qmeans"]]
    df_results_kmeans = df_results[df_results["kmeans"]]
    kmeans_palm_indexes = df_results_kmeans["palm"] == True
    df_results_kmeans_palm = df_results_kmeans[kmeans_palm_indexes]
    df_results_kmeans = df_results_kmeans[np.logical_not(kmeans_palm_indexes)]

    datasets = {
        # "Fashion Mnist": "--fashion-mnist",
        # "Mnist": "--mnist",
        "Blobs": "--blobs",
        # "LFW": "--lfw"
    }

    dataset_dim = {
        # "Fashion Mnist": 784,
        # "Mnist": 784,
        # "LFW": 1850,
        "Blobs": 2000
    }

    # shapes = {"Fashion Mnist": (28, 28),
    #             "Mnist": (28, 28),
    #             "LFW": (50, 37)}

    sparsity_values = sorted(set(df_results_qmeans["--sparsity-factor"]))
    nb_cluster_values = sorted(set(df_results_qmeans["--nb-cluster"]))


    tasks = [
        "assignation_mean_time",
                  "1nn_kmean_inference_time",
                  "nystrom_build_time",
                  "nystrom_inference_time",
                  "nystrom_sampled_error_reconstruction",
                  "traintime",
                  "1nn_kmean_accuracy",
    "batch_assignation_mean_time",
             "1nn_get_distance_time",
    "nystrom_svm_accuracy",
    "nystrom_svm_time"
    ]

    x_indices = np.arange(len(nb_cluster_values))

    y_axis_scale_by_task = {
        "assignation_mean_time": "linear",
        "1nn_kmean_inference_time": "linear",
        "nystrom_build_time": "linear",
        "nystrom_inference_time": "linear",
        "nystrom_sampled_error_reconstruction": "log",
        "traintime": "linear",
        "1nn_kmean_accuracy": "linear",
    "batch_assignation_mean_time": "linear",
        "1nn_get_distance_time": "linear",

    "nystrom_svm_accuracy": "linear",
    "nystrom_svm_time": "linear"
    }

    y_axis_label_by_task = {
        "assignation_mean_time": "time (s)",
        "1nn_kmean_inference_time": "time (s)",
        "nystrom_build_time": "time (s)",
        "nystrom_inference_time": "time (s)",
        "nystrom_sampled_error_reconstruction": "log(norm of difference)",
        "traintime": "time (s)",
        "1nn_kmean_accuracy": "accuracy",
        "batch_assignation_mean_time": "time (s)",
        "1nn_get_distance_time": "time(s)",
    "nystrom_svm_accuracy": "accuracy",
    "nystrom_svm_time": "time(s)"
    }

    nb_sample_batch_assignation_mean_time = set(df_results["--batch-assignation-time"].dropna().values.astype(int)).pop()
    nb_sample_assignation_mean_time = set(df_results["--assignation-time"].dropna().values.astype(int)).pop()
    nb_sample_nystrom = set(df_results["--nystrom"].dropna().values.astype(int)).pop()
    hierarchical_values = set(df_results["--hierarchical"].values)
    hierarchical_values = [False]

    other_1nn_methods = ["brute", "ball_tree", "kd_tree"]
    other_1nn_methods_names = {
        "brute": "Brute force search",
        "ball_tree": "Ball tree",
        "kd_tree": "KD tree"
    }

    color_by_sparsity = {
        2: "g",
        3: "b",
        5: "c"
    }

    for dataset_name in datasets:
        datasets_col = datasets[dataset_name]
        df_dataset_qmeans = df_results_qmeans[df_results_qmeans[datasets_col] != False]
        df_dataset_kmeans = df_results_kmeans[df_results_kmeans[datasets_col] != False]
        df_dataset_kmeans_palm = df_results_kmeans_palm[df_results_kmeans_palm[datasets_col] == True]
        nb_factors = [min(int(np.log2(nb_cluster)), int(np.log2(dataset_dim[dataset_name]))) for nb_cluster in nb_cluster_values]
        for hierarchical_value in hierarchical_values:
            df_hierarchical = df_dataset_qmeans[df_dataset_qmeans["--hierarchical"] == hierarchical_value]
            df_hierarchical_kmeans_palm = df_dataset_kmeans_palm[df_dataset_kmeans_palm["--hierarchical"] == hierarchical_value]

            # assignations_times_means_for_sparsity = []
            # assignations_times_std_for_sparsity = []
            for str_task in tasks:
                nb_kmeans_palm = len(sparsity_values) * 2
                # extra_bars = 1 + nb_kmeans_palm if "1nn_kmean" not in str_task else 3 + nb_kmeans_palm # for 1nn there are also the other methods (ball_tree, kd_tree, to plot)
                if "1nn_kmean" in str_task:
                    extra_bars = 3
                elif "nystrom_sampled_error_reconstruction" in str_task:
                    extra_bars = 1
                else:
                    extra_bars = 0

                bar_width = 0.9 / (len(sparsity_values) + extra_bars + 1)
                fig, ax = plt.subplots()
                plt.grid(zorder=-10)

                max_value_in_plot = 0
                bars = []
                for idx_sparsy_val, sparsy_val in enumerate(sparsity_values):
                    # Qmeans
                    ########
                    df_sparsy_val = df_hierarchical[df_hierarchical["--sparsity-factor"] == sparsy_val]
                    task_values = [df_sparsy_val[df_sparsy_val["--nb-cluster"] == clust_nbr][str_task] for clust_nbr in nb_cluster_values]

                    mean_task_values = [d.convert_objects(convert_numeric=True).dropna().mean() for d in task_values]
                    std_task_values = [d.convert_objects(convert_numeric=True).dropna().std() for d in task_values]
                    # assignations_times_means_for_sparsity.append(mean_time_values)
                    # assignations_times_std_for_sparsity.append(std_time_values)
                    bars.append(ax.bar(x_indices + bar_width * idx_sparsy_val, mean_task_values, bar_width, yerr=std_task_values,
                                       label='QK-means sparsity {}'.format(sparsy_val), zorder=10, color=color_by_sparsity[sparsy_val]))
                    max_value_in_plot = max(max_value_in_plot, max((np.array(mean_task_values) + np.array(std_task_values))))

                    # display number of parameters
                    for idx_bar, xcoor in enumerate(x_indices + bar_width * idx_sparsy_val):
                        nb_param = df_sparsy_val[df_sparsy_val["--nb-cluster"] == nb_cluster_values[idx_bar]]["nb_param_centroids"].mean()
                        ax.text(xcoor, mean_task_values[idx_bar] + std_task_values[idx_bar], ' {}'.format(int(round(nb_param))),
                                horizontalalignment='center',
                                verticalalignment='bottom',
                                rotation='vertical')

                    # kmeans palm
                    ##############
                    """
                    df_sparsy_val_kmeans_palm = df_hierarchical_kmeans_palm[df_hierarchical_kmeans_palm["--sparsity-factor"] == sparsy_val]
                    task_values_kmeans_palm = [df_sparsy_val_kmeans_palm[df_sparsy_val_kmeans_palm["--nb-cluster"] == clust_nbr][str_task] for clust_nbr in nb_cluster_values]
                    mean_task_values_kmeans_palm = [d.mean() for d in task_values_kmeans_palm]
                    std_task_values_kmeans_palm = [d.std() for d in task_values_kmeans_palm]
                    offset_from_qmeans = 2  # offset from qmeans = 2 because there is K means first
                    bars.append(ax.bar(x_indices + bar_width * (len(sparsity_values)-1 + offset_from_qmeans + idx_sparsy_val), mean_task_values_kmeans_palm, bar_width, yerr=std_task_values_kmeans_palm,
                                       label='Kmeans palm Sparsity {}'.format(sparsy_val)))
                    max_value_in_plot = max(max_value_in_plot, max((np.array(mean_task_values_kmeans_palm) + np.array(std_task_values_kmeans_palm))))

                    for idx_bar, xcoor in enumerate(x_indices + bar_width * (len(sparsity_values)-1 + idx_sparsy_val + offset_from_qmeans)):
                        nb_param = df_sparsy_val_kmeans_palm[df_sparsy_val_kmeans_palm["--nb-cluster"] == nb_cluster_values[idx_bar]]["nb_param_centroids"].mean()
                        ax.text(xcoor, mean_task_values_kmeans_palm[idx_bar] + std_task_values_kmeans_palm[idx_bar], '{}'.format(int(round(nb_param))),
                                horizontalalignment='center',
                                verticalalignment='bottom',
                                rotation='vertical')
                                """

                # Kmeans
                ########
                task_values_kmeans = [df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr][str_task] for clust_nbr in nb_cluster_values]
                mean_task_values_kmeans = [d.mean() for d in task_values_kmeans]
                std_task_values_kmeans = [d.std() for d in task_values_kmeans]
                offset_from_qmeans = 1  # offset from qmeans = 1 because directly after
                bars.append(ax.bar(x_indices + bar_width * (len(sparsity_values)-1+offset_from_qmeans), mean_task_values_kmeans, bar_width, yerr=std_task_values_kmeans,
                                   label='Kmeans', zorder=10, color="r"))
                max_value_in_plot = max(max_value_in_plot, max((np.array(mean_task_values_kmeans) + np.array(std_task_values_kmeans))))

                # display number of parameters
                for idx_bar, xcoor in enumerate(x_indices + bar_width * (idx_sparsy_val + offset_from_qmeans)):
                    nb_param = df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == nb_cluster_values[idx_bar]]["nb_param_centroids"].mean()
                    ax.text(xcoor, mean_task_values_kmeans[idx_bar] + std_task_values_kmeans[idx_bar], ' {}'.format(int(round(nb_param))),
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            rotation='vertical')

                # # for nearest neighbor: add other bars for brute, kdtree and balltree
                # if "1nn_kmean" in str_task:
                #     # offset_from_qmeans = 1 + len(sparsity_values) # offset from qmeans =3 because there are both kmeans first
                #     offset_from_qmeans = 1 # offset from qmeans =3 because there are both kmeans first
                #     for idx_other_1nn, str_other_1nn in enumerate(other_1nn_methods):
                #         str_task_special_1nn = str_task.replace("kmean", str_other_1nn)
                #         task_values_kmeans = [pd.to_numeric(df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr][str_task_special_1nn], errors="coerce") for clust_nbr in nb_cluster_values]
                #         mean_task_values_kmeans = [d.mean() for d in task_values_kmeans]
                #         std_task_values_kmeans = [d.std() for d in task_values_kmeans]
                #
                #         bars.append(ax.bar(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans + idx_other_1nn), mean_task_values_kmeans, bar_width, yerr=std_task_values_kmeans,
                #                            label=other_1nn_methods_names[str_other_1nn], zorder=10))
                #
                #         max_value_in_plot = max(max_value_in_plot, max(np.array(mean_task_values_kmeans) + np.array(std_task_values_kmeans)))
                #         # for idx_bar, xcoor in enumerate(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans + idx_other_1nn)):
                #         #     nb_param = df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == nb_cluster_values[idx_bar]]["nb_param_centroids"].mean()
                #         #     ax.text(xcoor, mean_task_values_kmeans[idx_bar] + std_task_values_kmeans[idx_bar], '{}'.format(int(round(nb_param))),
                #         #             horizontalalignment='center',
                #         #             verticalalignment='bottom',
                #         #             rotation='vertical')

                if "nystrom_sampled_error_reconstruction" in str_task:
                    offset_from_qmeans = 1 # offset from qmeans =1 because there are both kmeans first
                    str_task_special_1nn = "nystrom_sampled_error_reconstruction_uniform"
                    task_values_nystrom_uniform = [pd.to_numeric(df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr][str_task_special_1nn], errors="coerce") for clust_nbr in nb_cluster_values]
                    mean_task_values_nystrom_uniform = [d.mean() for d in task_values_nystrom_uniform]
                    std_task_values_nystrom_uniform = [d.std() for d in task_values_nystrom_uniform]

                    bars.append(ax.bar(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans), mean_task_values_nystrom_uniform, bar_width, yerr=std_task_values_nystrom_uniform,
                                       label="Uniform sampling", zorder=10, color="m"))

                    for idx_bar, xcoor in enumerate(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans)):
                        nb_param = df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == nb_cluster_values[idx_bar]]["nb_param_centroids"].mean()
                        ax.text(xcoor, mean_task_values_nystrom_uniform[idx_bar] + std_task_values_nystrom_uniform[idx_bar], ' {}'.format(int(round(nb_param))),
                                horizontalalignment='center',
                                verticalalignment='bottom',
                                rotation='vertical')

                    max_value_in_plot = max(max_value_in_plot, max(np.array(mean_task_values_nystrom_uniform) + np.array(std_task_values_nystrom_uniform)))



                title = '{}: {}'.format(dataset_name, str_task) + (" Hierarchical version" if hierarchical_value else "")
                if str_task == "batch_assignation_mean_time":
                    title += " size batch {}".format(nb_sample_batch_assignation_mean_time)
                elif str_task == "assignation_mean_time":
                    title += " nb samples {}".format(nb_sample_assignation_mean_time)
                elif "nystrom" in str_task:
                    title += " size matrix {}".format(nb_sample_nystrom)

                plt.yscale(y_axis_scale_by_task[str_task])

                # if "accuracy" in str_task and "LFW" not in dataset_name:
                #     ax.set_ylim(bottom=0.8)

                ax.set_ylim(top=max_value_in_plot * (1+1./2.8))

                ax.set_ylabel(y_axis_label_by_task[str_task])
                ax.set_xlabel('Number of clusters K')

                # ax.set_title("\n".join(wrap(title, 30)))
                ax.set_xticks(x_indices)

                xtick_labels = [str(nb_clust) + "({} factors)".format(nb_factors[idx_nb_clust]) for idx_nb_clust, nb_clust in enumerate(nb_cluster_values)]
                ax.set_xticklabels(xtick_labels)
                handles, labels = plt.gca().get_legend_handles_labels()
                ncol = len(labels) // 3
                # ax.legend(ncol=ncol, bbox_to_anchor=(0., -.3, 1., 0.102), mode="expand")
                ax.legend(ncol=2)
                fig.set_size_inches(8, 6)
                fig.tight_layout()
                # plt.show()
                plt.savefig(output_dir / title.replace(" ", "_").replace(":", ""))
