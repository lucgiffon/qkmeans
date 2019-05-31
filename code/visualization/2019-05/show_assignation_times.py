import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
from io import StringIO
from pandas.errors import EmptyDataError
from visualization.utils import get_dct_result_files_by_root

def build_df(dct_output_files_by_root, col_to_delete=[]):
    lst_df_results = []
    for root_name, dct_results in dct_output_files_by_root.items():
        result_file = src_result_dir / dct_results["results"]
        df_expe = pd.read_csv(result_file)
        lst_df_results.append(df_expe)

    df_results = pd.concat(lst_df_results)

    for c in col_to_delete:
        df_results = df_results.drop([c], axis=1)
    return df_results


def get_df(path):
    src_result_dir = pathlib.Path()
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
                     "1nn_ball_tree_inference_time",
                     "1nn_ball_tree_accuracy",
                     "1nn_brute_accuracy",
                     "1nn_brute_inference_time",
                     "1nn_kd_tree_accuracy",
                     "1nn_kd_tree_inference_time",
                     "--output-file",
                     "--output-file_centroidprinter",
                     "--output-file_objprinter",
                     "--output-file_resprinter"]

    df_results = build_df(dct_output_files_by_root, col_to_delete)
    return df_results


if __name__ == "__main__":

    df_results_qmeans = get_df("/home/luc/PycharmProjects/qalm_qmeans/results/2019-05/big_expe_less")
    df_results_kmeans = get_df("/home/luc/PycharmProjects/qalm_qmeans/results/2019-05/big_expe_less")


    datasets = {"Fashion Mnist": "--fashion-mnist",
                "Mnist": "--mnist"}

    sparsity_values = sorted(set(df_results_qmeans["--sparsity-factor"]))
    nb_cluster_values = sorted(set(df_results_qmeans["--nb-cluster"]))

    for dataset_name in datasets:
        datasets_col = datasets[dataset_name]
        df_dataset = df_results_qmeans[df_results_qmeans[datasets_col]]
        for hierarchical_value in [True, False]:
            df_hierarchical = df_dataset[df_dataset["--hierarchical"] == hierarchical_value]
            fig, ax = plt.subplots()

            x_indices = np.arange(len(sparsity_values))
            bar_width = 0.9 / (len(sparsity_values) + 1)

            # assignations_times_means_for_sparsity = []
            # assignations_times_std_for_sparsity = []
            bars = []
            for idx_sparsy_val, sparsy_val in enumerate(sparsity_values):
                df_sparsy_val = df_hierarchical[df_hierarchical["--sparsity-factor"] == sparsy_val]
                time_values = [df_sparsy_val[df_sparsy_val["--nb-cluster"] == clust_nbr]["assignation_mean_time"] for clust_nbr in nb_cluster_values]
                mean_time_values = [d.mean() for d in time_values]
                std_time_values = [d.std() for d in time_values]
                # assignations_times_means_for_sparsity.append(mean_time_values)
                # assignations_times_std_for_sparsity.append(std_time_values)
                bars.append(ax.bar(x_indices + bar_width*idx_sparsy_val, mean_time_values, bar_width, yerr=std_time_values,
                            label='Sparsity {}'.format(sparsy_val)))


            ax.set_ylabel('Time (s)')
            ax.set_xlabel('# clusters')
            ax.set_title('Assignation mean time')
            ax.set_xticks(x_indices + bar_width)
            ax.set_xticklabels(nb_cluster_values)
            ax.legend()
            fig.tight_layout()
            plt.show()
    # df_results_kmeans = build_df(dct_output_files_by_root, col_to_delete)



    a=1
