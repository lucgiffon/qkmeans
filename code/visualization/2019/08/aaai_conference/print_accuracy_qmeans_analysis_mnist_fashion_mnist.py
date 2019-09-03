import numpy as np
import pandas as pd
import pathlib

from visualization.utils import get_dct_result_files_by_root, build_df

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
    suf_path = "2019/07/qmeans_analysis_nystrom_mnist_fmnist"
    input_dir = "/home/luc/PycharmProjects/qalm_qmeans/results/" + suf_path
    output_dir = "/home/luc/PycharmProjects/qalm_qmeans/reports/figures/" + "2019/07/tech_report" + "/histogrammes"
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
    df_results["nystrom_inference_time"] = df_results["nystrom_inference_time"] * 1e3

    df_results_failure = df_results[df_results["failure"]]
    df_results = df_results[np.logical_not(df_results["failure"])]

    df_results_qmeans = df_results[df_results["qmeans"]]
    df_results_kmeans = df_results[df_results["kmeans"]]
    kmeans_palm_indexes = df_results_kmeans["palm"] == True
    df_results_kmeans_palm = df_results_kmeans[kmeans_palm_indexes]
    df_results_kmeans = df_results_kmeans[np.logical_not(kmeans_palm_indexes)]

    datasets = {
        "Fashion Mnist": "--fashion-mnist",
        "Mnist": "--mnist",
        # "Blobs": "--blobs",
        # "LFW": "--lfw"
    }

    dataset_dim = {
        "Fashion Mnist": 784,
        "Mnist": 784,
        # "LFW": 1850,
        # "Blobs": 2000
    }

    # shapes = {"Fashion Mnist": (28, 28),
    #             "Mnist": (28, 28),
    #             "LFW": (50, 37)}

    sparsity_values = sorted(set(df_results_qmeans["--sparsity-factor"]))
    nb_cluster_values = sorted(set(df_results_qmeans["--nb-cluster"]))


    tasks = [
          "1nn_kmean_accuracy",
          "nystrom_svm_accuracy",
        "nystrom_sampled_error_reconstruction",
        "nystrom_inference_time",
        "nystrom_sampled_error_reconstruction_uniform"
    ]

    x_indices = np.arange(len(nb_cluster_values))

    y_axis_scale_by_task = {
        "1nn_kmean_accuracy": "linear",
        "nystrom_svm_accuracy": "linear",
        "nystrom_sampled_error_reconstruction_uniform": "log",
        "nystrom_sampled_error_reconstruction": "log",
        "nystrom_inference_time": "linear",

    }

    y_axis_label_by_task = {
        "1nn_kmean_accuracy": "accuracy",
        "nystrom_svm_accuracy": "accuracy",
        "nystrom_sampled_error_reconstruction_uniform": "log(norm of difference)",
        "nystrom_sampled_error_reconstruction": "log(norm of difference)",
        "nystrom_inference_time": "time (ms)",
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
        print(dataset_name)
        datasets_col = datasets[dataset_name]
        df_dataset_qmeans = df_results_qmeans[df_results_qmeans[datasets_col] == True]
        df_dataset_kmeans = df_results_kmeans[df_results_kmeans[datasets_col] == True]
        df_dataset_kmeans_palm = df_results_kmeans_palm[df_results_kmeans_palm[datasets_col] == True]
        nb_factors = [min(int(np.log2(nb_cluster)), int(np.log2(dataset_dim[dataset_name]))) for nb_cluster in nb_cluster_values]
        for hierarchical_value in hierarchical_values:
            print("Hierarchical: {}".format(hierarchical_value))
            df_hierarchical = df_dataset_qmeans[df_dataset_qmeans["--hierarchical"] == hierarchical_value]
            df_hierarchical_kmeans_palm = df_dataset_kmeans_palm[df_dataset_kmeans_palm["--hierarchical"] == hierarchical_value]

            # assignations_times_means_for_sparsity = []
            # assignations_times_std_for_sparsity = []
            for str_task in tasks:
                nb_kmeans_palm = len(sparsity_values) * 2
                # extra_bars = 1 + nb_kmeans_palm if "1nn_kmean" not in str_task else 3 + nb_kmeans_palm # for 1nn there are also the other methods (ball_tree, kd_tree, to plot)

                max_value_in_plot = 0
                bars = []
                for idx_sparsy_val, sparsy_val in enumerate(sparsity_values):
                    print("Sparsity value: {}".format(sparsy_val))
                    # Qmeans
                    ########
                    df_sparsy_val = df_hierarchical[df_hierarchical["--sparsity-factor"] == sparsy_val]
                    task_values = [df_sparsy_val[df_sparsy_val["--nb-cluster"] == clust_nbr][str_task] for clust_nbr in nb_cluster_values]

                    mean_task_values = [d.convert_objects(convert_numeric=True).dropna().mean() for d in task_values]
                    std_task_values = [d.convert_objects(convert_numeric=True).dropna().std() for d in task_values]
                    # assignations_times_means_for_sparsity.append(mean_time_values)
                    # assignations_times_std_for_sparsity.append(std_time_values)

                    print("QK-means")
                    for i_val, _ in enumerate(mean_task_values):
                        mean_val = mean_task_values[i_val]
                        clust_nb = nb_cluster_values[i_val]
                        std_val = std_task_values[i_val]
                        print("Cluster nb: {} - {}: {} +/- {}".format(clust_nb, str_task, mean_val, std_val))

                # Kmeans
                ########
                print("K-means")
                task_values_kmeans = [df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr][str_task] for clust_nbr in nb_cluster_values]
                mean_task_values_kmeans = [d.mean() for d in task_values_kmeans]
                std_task_values_kmeans = [d.std() for d in task_values_kmeans]
                offset_from_qmeans = 1  # offset from qmeans = 1 because directly after

                for i_val, _ in enumerate(mean_task_values_kmeans):
                    mean_val = mean_task_values_kmeans[i_val]
                    clust_nb = nb_cluster_values[i_val]
                    std_val = std_task_values_kmeans[i_val]
                    print("Cluster nb: {} - {}: {} +/- {}".format(clust_nb, str_task, mean_val, std_val))

                # # for nearest neighbor: add other bars for brute, kdtree and balltree
                if "1nn_kmean" in str_task:
                    # offset_from_qmeans = 1 + len(sparsity_values) # offset from qmeans =3 because there are both kmeans first
                    for idx_other_1nn, str_other_1nn in enumerate(other_1nn_methods):
                        str_task_special_1nn = str_task.replace("kmean", str_other_1nn)
                        task_values_kmeans = [pd.to_numeric(df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr][str_task_special_1nn], errors="coerce") for clust_nbr in nb_cluster_values]
                        mean_task_values_kmeans = [d.mean() for d in task_values_kmeans]
                        std_task_values_kmeans = [d.std() for d in task_values_kmeans]
                        for i_val, _ in enumerate(mean_task_values_kmeans):
                            mean_val = mean_task_values_kmeans[i_val]
                            std_val = std_task_values_kmeans[i_val]
                            print("{}: {} - {}: {} +/- {}".format(str_other_1nn, clust_nb, str_task, mean_val, std_val))
