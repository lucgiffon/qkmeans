import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
import math
from io import StringIO
from matplotlib.ticker import MaxNLocator
from pandas.errors import EmptyDataError
from qkmeans.utils import logger
from visualization.utils import get_dct_result_files_by_root, build_df
from collections import OrderedDict
import matplotlib


font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 12
        }

matplotlib.rc('font', **font)


def get_objective_and_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir, old_filename_objective=True)

    col_to_delete = ["--initialization",
                     "--1-nn",
                     "--assignation-time",
                     "--help",
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

    dct_oarid_objective = {}
    for root_name, job_files in dct_output_files_by_root.items():
        objective_file_path = src_result_dir / job_files["objective"]
        loaded_objective = pd.read_csv(objective_file_path, skiprows=1)
        dct_oarid_objective[root_name] = loaded_objective

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return dct_oarid_objective, df_results


if __name__ == "__main__":
    # suf_path = "2019/06/qmeans_analysis_littledataset_fast"
    suf_path = "2019/07/qmeans_analysis_blobs_log2_clusters_bis"

    input_dir = "/home/luc/PycharmProjects/qalm_qmeans/results/" + suf_path
    output_dir = "/home/luc/PycharmProjects/qalm_qmeans/reports/figures/" + "2019/07/tech_report" + "/objectives"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dct_centroids, df_results = get_objective_and_df(input_dir)

    df_results_qmeans = df_results[df_results["qmeans"]]
    df_results_kmeans = df_results[df_results["kmeans"]]

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

    color_by_sparsity = {
        2: "g",
        3: "b",
        5: "c"
    }

    lst_sparsity_values = sorted(set(df_results_qmeans["--sparsity-factor"]))
    lst_nb_cluster_values = sorted(set(df_results_qmeans["--nb-cluster"]))

    x_indices = np.arange(len(lst_nb_cluster_values))

    for dataset_name in datasets:


        datasets_col = datasets[dataset_name]

        df_dataset_qmeans = df_results_qmeans[df_results_qmeans[datasets_col] != False]
        df_dataset_qmeans = df_dataset_qmeans[df_dataset_qmeans[datasets_col] != None]
        df_dataset_kmeans = df_results_kmeans[df_results_kmeans[datasets_col] != False]
        df_dataset_kmeans = df_dataset_kmeans[df_dataset_kmeans[datasets_col] != None]

        nb_factors = [min(int(np.log2(nb_cluster)), int(np.log2(dataset_dim[dataset_name]))) for nb_cluster in lst_nb_cluster_values]
        for idx_nb_clust, clust_nbr in enumerate(lst_nb_cluster_values):
            f, curr_ax = plt.subplots()
            # qmeans part #
            ###############
            df_nb_clust_qmeans = df_dataset_qmeans[df_dataset_qmeans["--nb-cluster"] == clust_nbr]

            for hierarchical_value in [False]:
                df_hierarchical = df_nb_clust_qmeans[df_nb_clust_qmeans["--hierarchical"] == hierarchical_value]

                for idx_sparsy_val, sparsy_val in enumerate(lst_sparsity_values):
                    # if sparsy_val == 1: continue
                    df_sparsy_val = df_hierarchical[df_hierarchical["--sparsity-factor"] == sparsy_val]

                    lst_objectives_seeds = []
                    for oar_id in df_sparsy_val["oar_id"]:
                        loss_values = dct_centroids[oar_id].values.squeeze()
                        lst_objectives_seeds.append(loss_values)
                    max_length = max(len(obj) for obj in lst_objectives_seeds)
                    for i, obj in enumerate(lst_objectives_seeds):
                        delta_length = int(max_length - len(obj))
                        if delta_length == 0: continue
                        padding = np.array([obj[-1]] * delta_length)
                        lst_objectives_seeds[i] = np.concatenate([obj, padding])

                    arr_objective_seeds = np.array(lst_objectives_seeds)
                    mean_objective = np.mean(arr_objective_seeds, axis=0).squeeze()
                    std_objective = np.std(arr_objective_seeds, axis=0).squeeze()
                    str_label = "{}QK-means sparsity {}".format("Hierarch. " if hierarchical_value else "", sparsy_val)
                    curr_ax.errorbar(np.arange(len(mean_objective)), mean_objective, yerr=std_objective, label=str_label,
                                     linestyle=":" if hierarchical_value else "-", color=color_by_sparsity[sparsy_val])

            # kmeans part #
            ###############
            df_nb_clust_kmeans = df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr]

            lst_objectives_seeds = []
            for oar_id in df_nb_clust_kmeans["oar_id"]:
                loss_values = dct_centroids[oar_id].values.squeeze()
                lst_objectives_seeds.append(loss_values)
            max_length = max(len(obj) for obj in lst_objectives_seeds)
            for i, obj in enumerate(lst_objectives_seeds):
                delta_length = int(max_length - len(obj))
                if delta_length == 0: continue
                padding = np.array([obj[-1]] * delta_length)
                lst_objectives_seeds[i] = np.concatenate([obj, padding])
            arr_objective_seeds = np.array(lst_objectives_seeds)
            mean_objective = np.mean(arr_objective_seeds, axis=0).squeeze()
            std_objective = np.std(arr_objective_seeds, axis=0).squeeze()
            str_label = "Kmeans"
            curr_ax.errorbar(np.arange(len(mean_objective)), mean_objective, yerr=std_objective, label=str_label, color="r")
            # curr_ax.set_yscale("symlog")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            # legend_ax = plt.subplot(2, 2, 4)  # type: plt.Axes
            # legend_ax.legend(by_label.values(), by_label.keys())
            # legend_ax.axis("off")
            # curr_ax.legend(bbox_to_anchor=(1, -0.1), ncol=2)
            plt.xlabel("Iteration number")
            plt.ylabel("Objective function value (logscale)")
            curr_ax.legend()
            curr_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # box = curr_ax.get_position()
            # curr_ax.set_position([box.x0, box.y0 + (box.height*0.25), box.width, box.height*0.8])
            title = "{} - # cluster: {}".format(dataset_name, clust_nbr)
            # plt.title(title)
            print(title)
            plt.grid()
            # plt.savefig(output_dir / title.replace(" ", "_").replace(":", "").replace("#", ""))

            plt.show()