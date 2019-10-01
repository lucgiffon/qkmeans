import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
import math
from io import StringIO
from pandas.errors import EmptyDataError
from qkmeans.utils import logger
from visualization.utils import get_dct_result_files_by_root, build_df


def get_centroids_and_df(path):
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

    dct_oarid_centroids = {}
    for root_name, job_files in dct_output_files_by_root.items():
        centroids_file_path = src_result_dir / job_files["centroids"]
        loaded_centroids = np.load(centroids_file_path, allow_pickle=True)
        print(loaded_centroids.shape)
        if len(loaded_centroids.shape) > 0:
            dct_oarid_centroids[root_name] = np.load(centroids_file_path, allow_pickle=True)
        else:
            dct_oarid_centroids[root_name] = loaded_centroids.item()

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return dct_oarid_centroids, df_results

def show_centroids_from_vector(arr_matrix_centroids, title):
    nb_centroids = arr_matrix_centroids.shape[0]
    nb_line = 3
    nb_centroid_by_line = math.ceil(nb_centroids / nb_line)

    # position = nb_line * 100 + nb_centroid_by_line * 10
    fig = plt.figure(figsize=(max(nb_centroid_by_line, 7), nb_line))

    for i in range(nb_centroids):
        tmp = i + 1
        print(nb_centroids, nb_line, nb_centroid_by_line, tmp)
        ax = plt.subplot(nb_line, nb_centroid_by_line, tmp)  # type: plt.Axes
        # line_idx = i // nb_centroid_by_line
        # col_idx = i % nb_centroid_by_line
        # idx = line_idx + col_idx
        reconstructed_centroid = arr_matrix_centroids[i].reshape((28, 28))
        ax.imshow(reconstructed_centroid)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle(title)
    # fig.tight_layout()

    fig.subplots_adjust(top=0.9, wspace=0, hspace=0)
    plt.savefig(output_dir / title.replace(" ", "_").replace(":", ""))
    # plt.show()


if __name__ == "__main__":

    dct_centroids, df_results = get_centroids_and_df("/home/luc/PycharmProjects/qalm_qmeans/results/2019-05/qmeans_analysis_littledataset_3_80_ghz_cpu")

    output_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/reports/figures/2019/05/qmeans_analysis_littledataset/centroids_display")

    df_results_qmeans = df_results[df_results["qmeans"]]
    df_results_kmeans = df_results[df_results["kmeans"]]

    datasets = {"Fashion Mnist": "--fashion-mnist",
                "Mnist": "--mnist"}

    dataset_dim = {"Fashion Mnist": 784,
                "Mnist": 784}

    lst_sparsity_values = sorted(set(df_results_qmeans["--sparsity-factor"]))
    lst_nb_cluster_values = sorted(set(df_results_qmeans["--nb-cluster"]))

    x_indices = np.arange(len(lst_nb_cluster_values))

    for dataset_name in datasets:
        datasets_col = datasets[dataset_name]

        # first deal with qmeans results
        ################################
        df_dataset_qmeans = df_results_qmeans[df_results_qmeans[datasets_col]]
        nb_factors = [min(int(np.log2(nb_cluster)), int(np.log2(dataset_dim[dataset_name]))) for nb_cluster in lst_nb_cluster_values]
        for hierarchical_value in [True, False]:
            df_hierarchical = df_dataset_qmeans[df_dataset_qmeans["--hierarchical"] == hierarchical_value]

            # fig, ax = plt.subplots()
            for idx_sparsy_val, sparsy_val in enumerate(lst_sparsity_values):
                df_sparsy_val = df_hierarchical[df_hierarchical["--sparsity-factor"] == sparsy_val]

                for idx_nb_clust, clust_nbr in enumerate(lst_nb_cluster_values):
                    df_nb_clust = df_sparsy_val[df_sparsy_val["--nb-cluster"] == clust_nbr]
                    # lst_objectives_seeds = []
                    for oar_id in df_nb_clust["oar_id"]:
                        # get centroid matrix for each seed (refered by the oar id)
                        flat_centroids = dct_centroids[oar_id].compute_product(return_array=True)
                        # lst_objectives_seeds.append(flat_centroids)
                        show_centroids_from_vector(flat_centroids, "{} Qmeans {} clusters sparsity factor {} {}{}".format(dataset_name, clust_nbr, sparsy_val, oar_id.split(".")[-1], " hierarchical" if hierarchical_value else ""))

        # then deal with kmeans results
        ###############################

        df_dataset_kmeans = df_results_kmeans[df_results_kmeans[datasets_col]]
        for idx_nb_clust, clust_nbr in enumerate(lst_nb_cluster_values):
            df_nb_clust = df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr]
            lst_centroids_seeds = []
            for oar_id in df_nb_clust["oar_id"]:
                # get centroid matrix for each seed (refered by the oar id)
                flat_centroids = dct_centroids[oar_id]
                # lst_objectives_seeds.append(flat_centroids)
                show_centroids_from_vector(flat_centroids, "{} Kmeans {} clusters {}".format(dataset_name, clust_nbr, oar_id.split(".")[-1]))
