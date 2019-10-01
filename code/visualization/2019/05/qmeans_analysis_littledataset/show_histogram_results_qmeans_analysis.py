import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
from io import StringIO
from pandas.errors import EmptyDataError
from qkmeans.utils import logger
from visualization.utils import get_dct_result_files_by_root

def build_df(path_results_dir, dct_output_files_by_root, col_to_delete=[]):
    lst_df_results = []
    for root_name, dct_results in dct_output_files_by_root.items():
        result_file = path_results_dir / dct_results["results"]
        df_expe = pd.read_csv(result_file)
        lst_df_results.append(df_expe)

    df_results = pd.concat(lst_df_results)

    for c in col_to_delete:
        df_results = df_results.drop([c], axis=1)
    return df_results


def get_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir)
    col_to_delete = ["--initialization",
                     "--1-nn",
                     "--assignation-time",
                     "--help",
                     "--blobs",
                     "--kddcup",
                     "--census",
                     "--nb-factors",
                     "--nb-iteration",
                     "--nb-iteration-palm",
                     "--nystrom",
                     "--plants",
                     "--verbose",
                     "--output-file",
                     "--output-file_centroidprinter",
                     "--output-file_objprinter",
                     "--output-file_resprinter"]

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return df_results


if __name__ == "__main__":

    df_results = get_df("/home/luc/PycharmProjects/qalm_qmeans/results/2019-05/qmeans_analysis_littledataset_3_80_ghz_cpu")

    output_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/reports/figures/2019/05/qmeans_analysis_littledataset")

    df_results_qmeans = df_results[df_results["qmeans"]]
    df_results_kmeans = df_results[df_results["kmeans"]]

    datasets = {"Fashion Mnist": "--fashion-mnist",
                "Mnist": "--mnist"}

    dataset_dim = {"Fashion Mnist": 784,
                "Mnist": 784}

    sparsity_values = sorted(set(df_results_qmeans["--sparsity-factor"]))
    nb_cluster_values = sorted(set(df_results_qmeans["--nb-cluster"]))


    tasks = ["assignation_mean_time",
                  "1nn_kmean_inference_time",
                  "nystrom_build_time",
                  "nystrom_inference_time",
                  "nystrom_sampled_error_reconstruction",
                  "traintime",
                  "1nn_kmean_accuracy"]

    x_indices = np.arange(len(nb_cluster_values))

    y_axis_scale_by_task = {
        "assignation_mean_time": "linear",
        "1nn_kmean_inference_time": "linear",
        "nystrom_build_time": "linear",
        "nystrom_inference_time": "linear",
        "nystrom_sampled_error_reconstruction": "log",
        "traintime": "linear",
        "1nn_kmean_accuracy": "linear"
    }

    y_axis_label_by_task = {
        "assignation_mean_time": "time (s)",
        "1nn_kmean_inference_time": "time (s)",
        "nystrom_build_time": "time (s)",
        "nystrom_inference_time": "time (s)",
        "nystrom_sampled_error_reconstruction": "log(norm of difference)",
        "traintime": "time (s)",
        "1nn_kmean_accuracy": "accuracy"
    }

    other_1nn_methods = ["brute", "ball_tree", "kd_tree"]

    for dataset_name in datasets:
        datasets_col = datasets[dataset_name]
        df_dataset_qmeans = df_results_qmeans[df_results_qmeans[datasets_col]]
        df_dataset_kmeans = df_results_kmeans[df_results_kmeans[datasets_col]]
        nb_factors = [min(int(np.log2(nb_cluster)), int(np.log2(dataset_dim[dataset_name]))) for nb_cluster in nb_cluster_values]
        for hierarchical_value in [True, False]:
            df_hierarchical = df_dataset_qmeans[df_dataset_qmeans["--hierarchical"] == hierarchical_value]

            # assignations_times_means_for_sparsity = []
            # assignations_times_std_for_sparsity = []
            for str_task in tasks:
                extra_bars = 1 if "1nn_kmean" not in str_task else 3 # for 1nn there are also the other methods (ball_tree, kd_tree, to plot)
                bar_width = 0.9 / (len(sparsity_values) + extra_bars + 1)

                fig, ax = plt.subplots()
                bars = []
                for idx_sparsy_val, sparsy_val in enumerate(sparsity_values):
                    df_sparsy_val = df_hierarchical[df_hierarchical["--sparsity-factor"] == sparsy_val]
                    task_values = [df_sparsy_val[df_sparsy_val["--nb-cluster"] == clust_nbr][str_task] for clust_nbr in nb_cluster_values]
                    mean_task_values = [d.mean() for d in task_values]
                    std_task_values = [d.std() for d in task_values]
                    # assignations_times_means_for_sparsity.append(mean_time_values)
                    # assignations_times_std_for_sparsity.append(std_time_values)
                    bars.append(ax.bar(x_indices + bar_width * idx_sparsy_val, mean_task_values, bar_width, yerr=std_task_values,
                                       label='Sparsity {}'.format(sparsy_val)))

                task_values_kmeans = [df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr][str_task] for clust_nbr in nb_cluster_values]
                mean_task_values_kmeans = [d.mean() for d in task_values_kmeans]
                std_task_values_kmeans = [d.std() for d in task_values_kmeans]
                bars.append(ax.bar(x_indices + bar_width * (idx_sparsy_val+1), mean_task_values_kmeans, bar_width, yerr=std_task_values_kmeans,
                                   label='Kmeans'))
                if "1nn_kmean" in str_task:
                    for idx_other_1nn, str_other_1nn in enumerate(other_1nn_methods):
                        str_task_special_1nn = str_task.replace("kmean", str_other_1nn)
                        task_values_kmeans = [pd.to_numeric(df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr][str_task_special_1nn], errors="coerce") for clust_nbr in nb_cluster_values]
                        mean_task_values_kmeans = [d.mean() for d in task_values_kmeans]
                        std_task_values_kmeans = [d.std() for d in task_values_kmeans]
                        bars.append(ax.bar(x_indices + bar_width * (idx_sparsy_val + 1 + (idx_other_1nn + 1)), mean_task_values_kmeans, bar_width, yerr=std_task_values_kmeans,
                                           label=str_other_1nn))


                title = '{}: {}'.format(dataset_name, str_task) + (" Hierarchical version" if hierarchical_value else "")
                plt.yscale(y_axis_scale_by_task[str_task])

                if "accuracy" in str_task:
                    ax.set_ylim(bottom=0.8)


                ax.set_ylabel(y_axis_label_by_task[str_task])
                ax.set_xlabel('# clusters')
                ax.set_title(title)
                ax.set_xticks(x_indices + bar_width)

                xtick_labels = [str(nb_clust) + "({} factors)".format(nb_factors[idx_nb_clust]) for idx_nb_clust, nb_clust in enumerate(nb_cluster_values)]
                ax.set_xticklabels(xtick_labels)
                ax.legend()
                fig.tight_layout()
                # plt.show()
                plt.savefig(output_dir / title.replace(" ", "_").replace(":", ""))
