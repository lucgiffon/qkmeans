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
    create_input_dir = lambda x: "/home/luc/PycharmProjects/qalm_qmeans/results/" + x
    # suf_path = "2019/10/5_6_new_expes"
    suf_path = "2019/10/5_6_new_expe_bis"
    input_dir = create_input_dir(suf_path)

    out_suf_path = "2020/01/ecml/0_0_efficient_nystrom_and_palm_on_kmeans_and_5_6_new_expe"
    output_dir = "/home/luc/PycharmProjects/qalm_qmeans/reports/figures/"+ out_suf_path + "/histogrammes"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    df_results = get_df(input_dir)
    df_results = df_results[np.logical_not(df_results["failure"])]
    df_results = df_results[df_results["--nb-cluster"].astype(int) != 1024]


    input_dir_efficient = create_input_dir("2020/01/0_0_efficient_nystrom_bis_bis")
    df_results_efficient = get_df(input_dir_efficient)

    suf_path_images_biggest = "2020/01/5_6_qmeans_analysis_caltech_mnist_fmnist_only_biggest_true"
    input_dir_images_biggest = create_input_dir(suf_path_images_biggest)
    df_results_image_biggest = get_df(input_dir_images_biggest)
    suf_path_images_others = "2020/01/5_6_qmeans_analysis_caltech_mnist_fmnist_others_true"
    input_dir_images_others = create_input_dir(suf_path_images_others)
    df_results_image_others = get_df(input_dir_images_others)

    df_results_image = pd.concat([df_results_image_biggest, df_results_image_others])

    df_results_image = df_results_image[np.logical_not(df_results_image["failure"])]
    df_results_image = df_results_image[df_results_image["--nb-cluster"].astype(int) != 1024]

    df_results = pd.concat([df_results, df_results_image])

    df_results_qmeans = df_results[df_results["qmeans"]]
    df_results_kmeans = df_results[df_results["kmeans"]]

    datasets = {
        "Fashion Mnist": "--fashion-mnist",
        "Mnist": "--mnist",
        # "Blobs": "--blobs",
        "Caltech": "--caltech256",
        "Kddcup04": "--kddcup04",
        "Kddcup99": "--kddcup99",
        "Census": "--census",
        "Plants": "--plants",
        "Breast Cancer": "--breast-cancer",
        "Coverage Type": "--covtype",
    }

    dataset_dim = {
        "Fashion Mnist": 784,
        "Mnist": 784,
        "Caltech": 2352,
        "Blobs": 2000,
        "Kddcup04": 74,
        "Kddcup99": 116,
        "Census": 68,
        "Plants": 70,
        "Breast Cancer": 30,
        "Coverage Type": 54,
    }

    # little_nb_clust = (8, 16, 32, 64)
    big_nb_clust = (8, 16, 32, 64, 128, 256)
    log2search = (8, 16, 32, 64, 128, 256, 512)
    lil_log2search = (8, 16, 32, 64)

    dataset_nb_cluster_value = {
        # "Fashion Mnist": lil_log2search,
        # "Mnist": lil_log2search,
        "Fashion Mnist": lil_log2search,
        "Mnist": lil_log2search,
        "Caltech": big_nb_clust,
        # "Blobs": big_nb_clust,
        "Kddcup04": log2search,
        "Kddcup99": log2search,
        "Census": log2search,
        "Coverage Type": log2search,
        "Plants": lil_log2search,
        "Breast Cancer": lil_log2search,
    }

    sparsity_values = sorted(set(df_results_qmeans["--sparsity-factor"]))
    # nb_cluster_values = sorted(set(df_results_qmeans["--nb-cluster"]))

    tasks = [
        # "assignation_mean_time",
        # "1nn_kmean_inference_time",
        # "nystrom_build_time",
        # "nystrom_inference_time",
        "nystrom_sampled_error_reconstruction",
        # "traintime",
        "1nn_kmean_accuracy",
        # "batch_assignation_mean_time",
        # "1nn_get_distance_time",
        "nystrom_svm_accuracy",
        # "nystrom_svm_time",
        "nb_param_centroids",
        "nb_flop_centroids"
    ]


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
        "nystrom_svm_time": "linear",
        "nb_param_centroids": "log",
        "nb_flop_centroids": "log"

    }

    y_axis_label_by_task = {
        "assignation_mean_time": "time (s)",
        "1nn_kmean_inference_time": "time (s)",
        "nystrom_build_time": "time (s)",
        "nystrom_inference_time": "time (s)",
        "nystrom_sampled_error_reconstruction": "log(error)",
        "traintime": "time (s)",
        "1nn_kmean_accuracy": "accuracy",
        "batch_assignation_mean_time": "time (s)",
        "1nn_get_distance_time": "time(s)",
        "nystrom_svm_accuracy": "accuracy",
        "nystrom_svm_time": "time(s)",
        "nb_param_centroids": "log(# non-zero values)",
        "nb_flop_centroids": "log(# FLOP)"
    }

    ncol_legend_by_task = {
        "assignation_mean_time": 2,
        "1nn_kmean_inference_time": 3,
        "nystrom_build_time": 3,
        "nystrom_inference_time": 3,
        "nystrom_sampled_error_reconstruction": 3,
        "traintime": 2,
        "1nn_kmean_accuracy":3,
        "batch_assignation_mean_time": 2,
        "1nn_get_distance_time": 3,
        "nystrom_svm_accuracy":3,
        "nystrom_svm_time": 3,
        "nb_param_centroids": 2,
        "nb_flop_centroids": 2
    }

    nb_sample_batch_assignation_mean_time = set(df_results["--batch-assignation-time"].dropna().values.astype(int)).pop()
    nb_sample_assignation_mean_time = set(df_results["--assignation-time"].dropna().values.astype(int)).pop()
    nb_sample_nystrom = set(df_results["--nystrom"].dropna().values.astype(int)).pop()
    hierarchical_value = False

    other_1nn_methods = ["brute", "ball_tree", "kd_tree"]

    other_nystrom_efficient_methods = [
        "nystrom_sampled_error_reconstruction_uniform",
        "nystrom_sampled_error_reconstruction_uop",
        "nystrom_sampled_error_reconstruction_uop_kmeans",
        # "nystrom_sampled_error_reconstruction_kmeans",
   ]
    other_nystrom_efficient_methods_accuracy = [
        "nystrom_svm_accuracy_uniform",
        "nystrom_svm_accuracy_uop",
        "nystrom_svm_accuracy_uop_kmeans",
        # "nystrom_svm_accuracy_kmeans",
   ]

    colors_nystrom = {
        "nystrom_sampled_error_reconstruction_uop": "#FF9933",
        "nystrom_sampled_error_reconstruction_uop_kmeans": "#FF6666",
        "nystrom_sampled_error_reconstruction_uniform": "m",
        "nystrom_sampled_error_reconstruction_kmeans": "k"
    }


    colors_nystrom_accuracy = {
        "nystrom_svm_accuracy_uop": "#FF9933",
        "nystrom_svm_accuracy_uop_kmeans": "#FF6666",
        "nystrom_svm_accuracy_uniform": "m",
        "nystrom_svm_accuracy_kmeans": "k"
    }



    other_1nn_methods_names = {
        "brute": "Brute force search",
        "ball_tree": "Ball tree",
        "kd_tree": "KD tree"
    }

    other_nystrom_efficient_methods_names = {
        "nystrom_sampled_error_reconstruction_uop": "Fast-Nys",
        "nystrom_sampled_error_reconstruction_uop_kmeans": "K Fast-nys",
        "nystrom_sampled_error_reconstruction_uniform": "Uniform",
        "nystrom_sampled_error_reconstruction_kmeans": "Kmeans2"
    }

    other_nystrom_efficient_methods_names_accuracy = {
        "nystrom_svm_accuracy_uop": "Fast-Nys",
        "nystrom_svm_accuracy_uop_kmeans": "K Fast-nys",
        "nystrom_svm_accuracy_uniform": "Uniform",
        "nystrom_svm_accuracy_kmeans": "Kmeans2",

    }

    color_by_sparsity = {
        2: "g",
        3: "b",
        5: "c"
    }

    for dataset_name in datasets:
        print("\n\n")
        print(dataset_name)

        datasets_col = datasets[dataset_name]
        if dataset_name in ("Fashion Mnist", "Mnist", "Kddcup04", "Kddcup99", "Census", "Coverage Type", "Plants", "Breast Cancer"):
            df_dataset_qmeans = df_results_qmeans[df_results_qmeans[datasets_col] != False]
            df_dataset_kmeans = df_results_kmeans[df_results_kmeans[datasets_col] != False]
            df_dataset_results_efficient = df_results_efficient[df_results_efficient[datasets_col] != False]
        else:
            df_dataset_qmeans = df_results_qmeans[df_results_qmeans[datasets_col] != "None"]
            df_dataset_kmeans = df_results_kmeans[df_results_kmeans[datasets_col] != "None"]
            df_dataset_results_efficient = df_results_efficient[df_results_efficient[datasets_col] != "None"]

        nb_cluster_values = dataset_nb_cluster_value[dataset_name]

        cluster_value_to_print_by_data = {
        "Fashion Mnist": -1,
        "Mnist": -1,
        "Caltech": -1,
        "Kddcup04": -1,
        "Kddcup99": -1,
        "Census": -1,
        "Coverage Type": -1,
        "Plants": -1,
        "Breast Cancer": -1,
        }


        x_indices = np.arange(len(nb_cluster_values))

        nb_factors = [max(int(np.log2(nb_cluster)), int(np.log2(dataset_dim[dataset_name]))) for nb_cluster in nb_cluster_values]

        for hierarchical_value in [True]:
            df_hierarchical = df_dataset_qmeans[df_dataset_qmeans["--hierarchical-init"] == hierarchical_value]

            for str_task in tasks:
                print(str_task)
                nb_kmeans_palm = len(sparsity_values) * 2
                # extra_bars = 1 + nb_kmeans_palm if "1nn_kmean" not in str_task else 3 + nb_kmeans_palm # for 1nn there are also the other methods (ball_tree, kd_tree, to plot)
                if "1nn_kmean" in str_task:
                    extra_bars = 5
                elif "nystrom_sampled_error_reconstruction" in str_task:
                    extra_bars = 5 # uniform, uop uniform, uop kmeans
                elif "nystrom_svm_accuracy" in str_task:
                    extra_bars = 5 # uniform, uop uniform, uop kmeans
                else:
                    extra_bars = 2 # for extra space between expe

                if str_task != "nb_flop_centroids" and (df_hierarchical[str_task] == "None").all():
                    # task is not defined for this dataset
                    continue


                bar_width = 0.9 / (len(sparsity_values) + extra_bars + 1)
                fig, ax = plt.subplots()
                plt.grid(zorder=-10)

                max_value_in_plot = 0
                bars = []
                for idx_sparsy_val, sparsy_val in enumerate(sparsity_values):
                    # Qmeans
                    ########
                    df_sparsy_val = df_hierarchical[df_hierarchical["--sparsity-factor"] == sparsy_val]
                    if str_task == "nb_flop_centroids":
                        # times 2 for the multiplications then additions
                        task_values = [df_sparsy_val[df_sparsy_val["--nb-cluster"] == clust_nbr]["nb_param_centroids"] * 2 for clust_nbr in nb_cluster_values]
                    else:
                        task_values = [df_sparsy_val[df_sparsy_val["--nb-cluster"] == clust_nbr][str_task] for clust_nbr in nb_cluster_values]

                    try:
                        mean_task_values = [pd.to_numeric(d).dropna().mean() for d in task_values]
                    except Exception as e:
                        raise e
                    std_task_values = [pd.to_numeric(d).dropna().std() for d in task_values]
                    # assignations_times_means_for_sparsity.append(mean_time_values)
                    # assignations_times_std_for_sparsity.append(std_time_values)
                    bars.append(ax.bar(x_indices + bar_width * idx_sparsy_val, mean_task_values, bar_width, yerr=std_task_values,
                                       label='QK-means sparsity {}'.format(sparsy_val), zorder=10, color=color_by_sparsity[sparsy_val]))
                    max_value_in_plot = max(max_value_in_plot, max((np.array(mean_task_values) + np.array(std_task_values))))

                    if sparsy_val == 3:
                        print("qkmeans; sparsity {}; clusternbr {}".format(sparsy_val, nb_cluster_values[cluster_value_to_print_by_data[dataset_name]]))
                        print(np.around(mean_task_values[cluster_value_to_print_by_data[dataset_name]], 4))

                    # display number of parameters
                    # for idx_bar, xcoor in enumerate(x_indices + bar_width * idx_sparsy_val):
                    #     try:
                    #         nb_param = df_sparsy_val[df_sparsy_val["--nb-cluster"] == nb_cluster_values[idx_bar]]["nb_param_centroids"].mean()
                    #         ax.text(xcoor, mean_task_values[idx_bar] + std_task_values[idx_bar], ' {}'.format(int(round(nb_param))),
                    #                 horizontalalignment='center',
                    #                 verticalalignment='bottom',
                    #                 rotation='vertical')
                    #     except Exception as e:
                    #         print("there is a pb")
                    #         raise e


                # Kmeans
                ########
                if str_task == "nb_flop_centroids":
                    # times 2 for the multiplications then additions
                    task_values_kmeans = [df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr]["nb_param_centroids"] * 2 for clust_nbr in nb_cluster_values]
                else:
                    task_values_kmeans = [df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr][str_task] for clust_nbr in nb_cluster_values]
                try:
                    mean_task_values_kmeans = [d.mean() for d in task_values_kmeans]
                except Exception as e:
                    raise e
                std_task_values_kmeans = [d.std() for d in task_values_kmeans]
                print("kmeans; clusternbr {}".format(nb_cluster_values[cluster_value_to_print_by_data[dataset_name]]))
                print(np.around(mean_task_values_kmeans[cluster_value_to_print_by_data[dataset_name]], 4))
                offset_from_qmeans = 1  # offset from qmeans = 1 because directly after
                bars.append(ax.bar(x_indices + bar_width * (len(sparsity_values)-1+offset_from_qmeans), mean_task_values_kmeans, bar_width, yerr=std_task_values_kmeans,
                                   label='Kmeans', zorder=10, color="r"))
                max_value_in_plot = max(max_value_in_plot, max((np.array(mean_task_values_kmeans) + np.array(std_task_values_kmeans))))

                # display number of parameters
                # for idx_bar, xcoor in enumerate(x_indices + bar_width * (idx_sparsy_val + offset_from_qmeans)):
                #     nb_param = df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == nb_cluster_values[idx_bar]]["nb_param_centroids"].mean()
                #     ax.text(xcoor, mean_task_values_kmeans[idx_bar] + std_task_values_kmeans[idx_bar], ' {}'.format(int(round(nb_param))),
                #             horizontalalignment='center',
                #             verticalalignment='bottom',
                #             rotation='vertical')

                # for nearest neighbor: add other bars for brute, kdtree and balltree
                if "1nn_kmean" in str_task:
                    # offset_from_qmeans = 1 + len(sparsity_values) # offset from qmeans =3 because there are both kmeans first
                    offset_from_qmeans = 1 # offset from qmeans =3 because there are both kmeans first
                    for idx_other_1nn, str_other_1nn in enumerate(other_1nn_methods):
                        str_task_special_1nn = str_task.replace("kmean", str_other_1nn)
                        task_values_kmeans = [pd.to_numeric(df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == clust_nbr][str_task_special_1nn], errors="coerce") for clust_nbr in nb_cluster_values]
                        mean_task_values_kmeans = [d.mean() for d in task_values_kmeans]
                        std_task_values_kmeans = [d.std() for d in task_values_kmeans]
                        if len(mean_task_values_kmeans):
                            print(str_other_1nn)
                            print(np.around(mean_task_values_kmeans[cluster_value_to_print_by_data[dataset_name]], 4))
                        bars.append(ax.bar(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans + idx_other_1nn), mean_task_values_kmeans, bar_width, yerr=std_task_values_kmeans,
                                           label=other_1nn_methods_names[str_other_1nn], zorder=10))

                        max_value_in_plot = max(max_value_in_plot, max(np.array(mean_task_values_kmeans) + np.array(std_task_values_kmeans)))
                        # for idx_bar, xcoor in enumerate(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans + idx_other_1nn)):
                        #     nb_param = df_dataset_kmeans[df_dataset_kmeans["--nb-cluster"] == nb_cluster_values[idx_bar]]["nb_param_centroids"].mean()
                        #     ax.text(xcoor, mean_task_values_kmeans[idx_bar] + std_task_values_kmeans[idx_bar], '{}'.format(int(round(nb_param))),
                        #             horizontalalignment='center',
                        #             verticalalignment='bottom',
                        #             rotation='vertical')

                if "nystrom_sampled_error_reconstruction" in str_task:
                    print("nystrom_sampled_error_reconstruction")
                    offset_from_qmeans = 1 # offset from qmeans = 1 because there is kmeans first

                    for idx_other_nystrom, str_other_nystrom in enumerate(other_nystrom_efficient_methods):
                        # str_task_special_1nn = "nystrom_sampled_error_reconstruction_uniform"
                        task_values_other_nystrom = [pd.to_numeric(df_dataset_results_efficient[df_dataset_results_efficient["--nb-landmarks"] == clust_nbr][str_other_nystrom], errors="coerce") for clust_nbr in
                                                       nb_cluster_values]
                        mean_task_values_nystrom_uniform = [d.mean() for d in task_values_other_nystrom]
                        std_task_values_nystrom_uniform = [d.std() for d in task_values_other_nystrom]
                        print(str_other_nystrom)
                        print(np.around(mean_task_values_nystrom_uniform[cluster_value_to_print_by_data[dataset_name]],4))
                        bars.append(ax.bar(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans + idx_other_nystrom), mean_task_values_nystrom_uniform, bar_width, yerr=std_task_values_nystrom_uniform,
                                           label=other_nystrom_efficient_methods_names[str_other_nystrom], zorder=10, color=colors_nystrom[str_other_nystrom]))

                        # for idx_bar, xcoor in enumerate(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans + idx_other_nystrom)):
                        #     nb_param = int(df_dataset_results_efficient[df_dataset_results_efficient["--nb-landmarks"] == nb_cluster_values[idx_bar]]["--seed"].mean() * dataset_dim[dataset_name])
                        #     ax.text(xcoor, mean_task_values_nystrom_uniform[idx_bar] + std_task_values_nystrom_uniform[idx_bar], ' {}'.format(int(round(nb_param))),
                        #             horizontalalignment='center',
                        #             verticalalignment='bottom',
                        #             rotation='vertical')

                        max_value_in_plot = max(max_value_in_plot, max(np.array(mean_task_values_nystrom_uniform) + np.array(std_task_values_nystrom_uniform)))

                if "nystrom_svm_accuracy" in str_task:
                    print("nystrom_svm_accuracy")
                    offset_from_qmeans = 1  # offset from qmeans = 1 because there is kmeans first

                    for idx_other_nystrom, str_other_nystrom in enumerate(other_nystrom_efficient_methods_accuracy):
                        # str_task_special_1nn = "nystrom_sampled_error_reconstruction_uniform"
                        task_values_other_nystrom = [pd.to_numeric(df_dataset_results_efficient[df_dataset_results_efficient["--nb-landmarks"] == clust_nbr][str_other_nystrom], errors="coerce") for
                                                     clust_nbr in
                                                     nb_cluster_values]
                        mean_task_values_nystrom_uniform = [d.mean() for d in task_values_other_nystrom]
                        std_task_values_nystrom_uniform = [d.std() for d in task_values_other_nystrom]
                        print(str_other_nystrom)
                        print(np.around(mean_task_values_nystrom_uniform[cluster_value_to_print_by_data[dataset_name]], 4))
                        bars.append(ax.bar(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans + idx_other_nystrom), mean_task_values_nystrom_uniform, bar_width,
                                           yerr=std_task_values_nystrom_uniform,
                                           label=other_nystrom_efficient_methods_names_accuracy[str_other_nystrom], zorder=10, color=colors_nystrom_accuracy[str_other_nystrom]))

                        # for idx_bar, xcoor in enumerate(x_indices + bar_width * (len(sparsity_values) + offset_from_qmeans + idx_other_nystrom)):
                        #     nb_param = int(df_dataset_results_efficient[df_dataset_results_efficient["--nb-landmarks"] == nb_cluster_values[idx_bar]]["--seed"].mean() * dataset_dim[dataset_name])
                        #     ax.text(xcoor, mean_task_values_nystrom_uniform[idx_bar] + std_task_values_nystrom_uniform[idx_bar], ' {}'.format(int(round(nb_param))),
                        #             horizontalalignment='center',
                        #             verticalalignment='bottom',
                        #             rotation='vertical')

                        max_value_in_plot = max(max_value_in_plot, max(np.array(mean_task_values_nystrom_uniform) + np.array(std_task_values_nystrom_uniform)))

                title = '{}: {}'.format(dataset_name, str_task) + (" Hierarchical version" if hierarchical_value else "")
                if str_task == "batch_assignation_mean_time":
                    title += " size batch {}".format(nb_sample_batch_assignation_mean_time)
                elif str_task == "assignation_mean_time":
                    title += " nb samples {}".format(nb_sample_assignation_mean_time)
                elif "nystrom" in str_task:
                    title += " size matrix {}".format(nb_sample_nystrom)

                plt.yscale(y_axis_scale_by_task[str_task])

                # if "accuracy" in str_task:
                #     ax.set_ylim(top=max_value_in_plot * 1.1)
                # else:
                #     ax.set_ylim(top=max_value_in_plot * (1+1./1.3))

                ax.set_ylim(top=max_value_in_plot * 1.1)

                ax.set_ylabel(y_axis_label_by_task[str_task])
                ax.set_xlabel('Number of clusters K (number of factors)' if "nystrom" not in str_task else "Number of landmarks")

                ax.set_xticks(x_indices)

                xtick_labels = [str(nb_clust) + "({})".format(nb_factors[idx_nb_clust]) for idx_nb_clust, nb_clust in enumerate(nb_cluster_values)]
                ax.set_xticklabels(xtick_labels)
                handles, labels = plt.gca().get_legend_handles_labels()
                # ncol = len(labels) // 3
                # ax.legend(ncol=ncol_legend_by_task[str_task], bbox_to_anchor=(0., 1.2, 1., 0.102), mode="expand")
                if "nys" not in str_task and "1nn" not in str_task:
                    ax.legend(ncol=ncol_legend_by_task[str_task])

                fig.set_size_inches(10.65,  4.)
                fig.tight_layout()
                # plt.show()
                plt.savefig(output_dir / title.replace(" ", "_").replace(":", ""))
