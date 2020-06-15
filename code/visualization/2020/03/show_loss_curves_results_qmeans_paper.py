from collections import defaultdict

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
import plotly.io as pio

import plotly.graph_objects as go
import numpy as np

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)
font = {'family': 'normal',
        # 'weight' : 'bold',
        'size': 16
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


def fill_objective_values(objectives, max_len_objectives):
    filled_objectives = [obj[-1] * np.ones(max_len_objectives) for obj in objectives]
    for i_obj, obj in enumerate(filled_objectives):
        obj[:len(objectives[i_obj])] = objectives[i_obj]

    return filled_objectives


color_by_sparsity = {
    2: (30, 144, 255), # dodgerblue
    3: (0, 0, 255), # blue
    5: (128, 0, 128) # purple
}
color_by_n_iter = {
    50:  (30, 144, 255), # dodgerblue
    200: (0, 0, 255), # blue
    300: (128, 0, 128) # purple
}


color_by_init = {
    "kmeans++": (0, 0, 255), # green
    "uniform_sampling": (255, 0, 0) # ref
}

tpl_transparency = (0.2,)

dct_str_init = {
    "kmeans++": "Kmeans ++",
    "uniform_sampling": "Uniform"
}

dash_by_init = {
    "kmeans++": "solid",
    "uniform_sampling": "dot"
}

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
    #
    # # suf_path = "2020/03/6_7_qmeans_all"
    # input_dir = results_dir / suf_path
    # processed_csv = input_dir / "processed.csv"
    # input_dir_small = "2020/06/7_8_qmeans_more_iter_small_datasets"
    #
    # df_results = pd.read_csv(processed_csv)
    # df_results = df_results.fillna("None")

    figures_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/reports/figures/")
    output_dir = figures_dir / suf_path / "curves"

    datasets = set(df_results["dataset"].values)
    sparsity_values = set(df_results["sparsity-factor"].values)
    sparsity_values.remove("None")
    nb_iter_palm = set(df_results["nb-iteration-palm"].values)

    # hierarchical_values = set(df_results["hierarchical-init"])
    nb_iter_kmeans = max(set(df_results["nb-iteration"]))
    i = 0

    # df_results = df_results[df_results["initialization"] == "kmeans++"]
    # df_results = df_results[df_results["dataset"] == "Coverage Type"]
    # df_results = df_results[df_results["dataset"] == "Caltech256 32"]
    # df_results = df_results[df_results["model"] == "Kmeans"]
    df_results = df_results[df_results["nb-iteration-palm"] != 100]


    for data in datasets:
        df_data = df_results[df_results["dataset"] == data]
        nb_clusters = sorted(set(df_data["nb-cluster"].values))
        for nb_cluster in nb_clusters:
            # if i==1: exit()
            # i=1
            df_nb_cluster = df_data[df_data["nb-cluster"] == nb_cluster]

            # these dicts will contain all figures
            dct_sparsity_figure = defaultdict(lambda: go.Figure())
            title_fig_sparsity_figures = "{} nb clust {} sparsity {} {}"
            dct_niter_figure = defaultdict(lambda: go.Figure())
            title_fig_niter_figures = "{} {} nb_iter {} {}"
            dct_init_figure = defaultdict(lambda: go.Figure())
            title_fig_init_figures = "{} {} sparsity: {}; n iter palm {}"

            ###########
            # QKMEANS #
            ###########
            df_qkmeans = df_nb_cluster[df_nb_cluster["model"] == "QKmeans"]
            for hierarchical_value in [True]:
                df_hierarchical = df_qkmeans[df_qkmeans["hierarchical-init"] == hierarchical_value]
                for sparsity_value in sorted(sparsity_values):
                    df_sparsity = df_hierarchical[df_hierarchical["sparsity-factor"] == sparsity_value]

                    for nb_iter in sorted(nb_iter_palm):
                        df_iter = df_sparsity[df_sparsity["nb-iteration-palm"] == nb_iter]

                        init_schemes = set(df_iter["initialization"].values)
                        for init in init_schemes:
                            df_init = df_iter[df_iter["initialization"] == init]
                            objectives = [np.load(path, allow_pickle=True)["qmeans_objective"][1] for path in df_init["path-objective"]]
                            objectives = fill_objective_values(objectives, nb_iter_kmeans)  # remplis les objectifs pour qu'ils fassent tous la même taille

                            mean_objectives = np.mean(objectives, axis=0)
                            std_objectives = np.std(objectives, axis=0)
                            std_objectives_upper = list(mean_objectives + std_objectives)
                            std_objectives_lower = list(mean_objectives - std_objectives)
                            x_axis_vals = list(np.arange(len(mean_objectives)))

                            #########################
                            # FIGURE SPARSITY FIXED #
                            #########################
                            name_fig = title_fig_sparsity_figures.format(data, nb_cluster, sparsity_value, init)
                            dct_sparsity_figure[name_fig].add_trace(go.Scatter(
                                x=np.arange(len(mean_objectives)),
                                y=mean_objectives,
                                # error_y=dict(
                                #     type='data',
                                #     # value of error bar given in data coordinates
                                #     array=std_objectives,
                                #     visible=True
                                # ),
                                line=dict(color="rgb{}".format(color_by_n_iter[int(nb_iter)])),
                                mode='lines',
                                name='QKmeans {} iterations'.format(nb_iter)))

                            dct_sparsity_figure[name_fig].add_trace(go.Scatter(
                                x=x_axis_vals + x_axis_vals[::-1],
                                y=std_objectives_upper + std_objectives_lower[::-1],
                                fill='toself',
                                showlegend=False,
                                fillcolor='rgba{}'.format(color_by_n_iter[int(nb_iter)] + tpl_transparency),
                                line_color='rgba(255,255,255,0)',
                                name='QKmeans {} iterations'.format(int(nb_iter)))
                            )

                            # ###########################
                            # # FIGURE NITER PALM FIXED #
                            # ###########################
                            name_fig_niter_fixed = title_fig_niter_figures.format(data, nb_cluster, nb_iter, init)
                            dct_niter_figure[name_fig_niter_fixed].add_trace(go.Scatter(x=np.arange(len(mean_objectives)),
                                                                                       y=mean_objectives,
                                                                                       # error_y=dict(
                                                                                       #     type='data',  # value of error bar given in data coordinates
                                                                                       #     array=std_objectives,
                                                                                       #     visible=True,
                                                                                       #     # color=color_by_sparsity[int(sparsity_value)]
                                                                                       # ),
                                                                                       line=dict(color="rgb{}".format(color_by_sparsity[int(sparsity_value)])),
                                                                                       mode='lines',
                                                                                       name='QKmeans sparsity level {}'.format(int(sparsity_value))))

                            dct_niter_figure[name_fig_niter_fixed].add_trace(go.Scatter(
                                x=x_axis_vals + x_axis_vals[::-1],
                                y=std_objectives_upper + std_objectives_lower[::-1],
                                fill='toself',
                                showlegend=False,
                                fillcolor='rgba{}'.format(color_by_sparsity[int(sparsity_value)] + tpl_transparency),
                                line_color='rgba(255,255,255,0)',
                                name='QKmeans sparsity level {}'.format(int(sparsity_value))
                            ))
                            #
                            ##############################
                            # FIGURE VARYING INIT METHOD #
                            ##############################
                            dct_init_figure[title_fig_init_figures.format(data, nb_cluster, str(sparsity_value), str(nb_iter))].add_trace(go.Scatter(x=np.arange(len(mean_objectives)),
                                                                                                                                                     y=mean_objectives,
                                                                                                                                                     # error_y=dict(
                                                                                                                                                     #     type='data',
                                                                                                                                                     #     # value of error bar given in data coordinates
                                                                                                                                                     #     array=std_objectives,
                                                                                                                                                     #     visible=True
                                                                                                                                                     # ),
                                                                                                                                                     line=dict(color="rgb{}".format((0, 0, 255)), dash=dash_by_init[init]),
                                                                                                                                                     mode='lines',
                                                                                                                                                     name='QKmeans {}'.format(dct_str_init[init])))
                            dct_init_figure[title_fig_init_figures.format(data, nb_cluster, str(sparsity_value), str(nb_iter))].add_trace(go.Scatter(
                                x=x_axis_vals + x_axis_vals[::-1],
                                y=std_objectives_upper + std_objectives_lower[::-1],
                                fill='toself',
                                showlegend=False,
                                fillcolor='rgba{}'.format((0, 0, 255) + tpl_transparency),
                                line_color='rgba(255,255,255,0)',
                                name='QKmeans {}'.format(dct_str_init[init])
                            ))

            ###########
            # KMEANS #
            ###########
            df_kmeans = df_nb_cluster[df_nb_cluster["model"] == "Kmeans"]
            init_schemes = set(df_kmeans["initialization"].values)
            for init in init_schemes:
                df_init = df_kmeans[df_kmeans["initialization"] == init]
                objectives = [np.load(path, allow_pickle=True)["kmeans_objective"][1] for path in df_init["path-objective"]]
                objectives = fill_objective_values(objectives, nb_iter_kmeans)  # remplis les objectifs avec leur meilleur valeur pour qu'ils fassent tous la même taille

                for obj in objectives:
                    lst_bool_gt = [obj[i] >= obj[i+1] for i in range(len(obj) - 1) ]
                    try:
                        assert all(lst_bool_gt)
                    except:
                        print("objective values of kmeans are not all decreasing")
                mean_objectives = np.mean(objectives, axis=0)
                std_objectives = np.std(objectives, axis=0)
                std_objectives_upper = list(mean_objectives + std_objectives)
                std_objectives_lower = list(mean_objectives - std_objectives)
                x_axis_vals = list(np.arange(len(mean_objectives)))

                ##################################################
                # FIGURE PALM IS VARYING THEN KMEANS IS CONSTANT #
                ##################################################
                for dct_figs in [dct_sparsity_figure, dct_niter_figure]:
                    for title, fig in dct_figs.items():
                        if init in title:
                            fig.add_trace(go.Scatter(x=np.arange(len(mean_objectives)),
                                                     y=mean_objectives,
                                                     # error_y=dict(
                                                     #     type='data',  # value of error bar given in data coordinates
                                                     #     array=std_objectives,
                                                     #     visible=True
                                                     # ),
                                                     line=dict(color="rgb(255, 0, 0)", dash="dot"),
                                                     mode='lines',
                                                     name='Kmeans'))
                            fig.add_trace(go.Scatter(
                                x=x_axis_vals + x_axis_vals[::-1],
                                y=std_objectives_upper + std_objectives_lower[::-1],
                                fill='toself',
                                showlegend=False,
                                fillcolor='rgba{}'.format((255, 0, 0) + tpl_transparency),
                                line_color='rgba(255,255,255,0)',
                             name='Kmeans'
                        ))


                ##############################
                # Figure Varying INIT METHOD #
                ##############################
                for title, fig in dct_init_figure.items():
                    fig.add_trace(go.Scatter(x=np.arange(len(mean_objectives)),
                                             y=mean_objectives,
                                             # error_y=dict(
                                             #     type='data',  # value of error bar given in data coordinates
                                             #     array=std_objectives,
                                             #     visible=True
                                             # ),
                                             line=dict(color="rgb{}".format((255, 0, 0)), dash=dash_by_init[init]),
                                             mode='lines',
                                             name='Kmeans {}'.format(dct_str_init[init])))

                    fig.add_trace(go.Scatter(
                        x=x_axis_vals + x_axis_vals[::-1],
                        y=std_objectives_upper + std_objectives_lower[::-1],
                        fill='toself',
                        showlegend=False,
                        fillcolor='rgba{}'.format((255, 0, 0) + tpl_transparency),
                        line_color='rgba(255,255,255,0)',
                        name='Kmeans {}'.format(dct_str_init[init])
                    ))

            # ##################
            # # KMEANS  + PALM #
            # ##################
            # df_kmeans_palm = df_nb_cluster[df_nb_cluster["model"] == "Kmeans + Palm"]
            #
            # for hierarchical_value in [True]:
            #     df_hierarchical = df_kmeans_palm[df_kmeans_palm["hierarchical-inside"] == hierarchical_value]
            #     for sparsity_value in sparsity_values:
            #         df_sparsity = df_hierarchical[df_hierarchical["sparsity-factor"] == sparsity_value]
            #         for nb_iter in nb_iter_palm:
            #             df_iter = df_sparsity[df_sparsity["nb-iteration-palm"] == nb_iter]
            #             for init in init_schemes:
            #                 df_init = df_iter[df_iter["initialization"] == init]
            #
            #                 objectives = [df_init["final-objective-value"] for _ in range(nb_iter_kmeans)]
            #                 # objectives = fill_objective_values(objectives, nb_iter_kmeans)  # remplis les objectifs pour qu'ils fassent tous la même taille
            #
            #                 mean_objectives = np.mean(objectives, axis=1)
            #                 std_objectives = np.std(objectives, axis=1)
            #                 std_objectives_upper = list(mean_objectives + std_objectives)
            #                 std_objectives_lower = list(mean_objectives - std_objectives)
            #                 x_axis_vals = list(np.arange(len(mean_objectives)))
            #
            #                 #########################
            #                 # FIGURE SPARSITY FIXED #
            #                 #########################
            #                 dct_sparsity_figure[title_fig_sparsity_figures.format(data, nb_cluster, sparsity_value, init)].add_trace(go.Scatter(x=np.arange(len(mean_objectives)),
            #                                                                                                                                     y=mean_objectives,
            #                                                                                                                                     # error_y=dict(
            #                                                                                                                                     #     type='data',  # value of error bar given in data coordinates
            #                                                                                                                                     #     array=std_objectives,
            #                                                                                                                                     #     visible=True
            #                                                                                                                                     # ),
            #                                                                                                                                     line=dict(color="rgb{}".format(color_by_n_iter[int(nb_iter)]),
            #                                                                                                                                               dash="dot"),
            #                                                                                                                                     mode='lines',
            #                                                                                                                                     name='Kmeans + PALM {}'.format(nb_iter)))
            #
            #                 dct_sparsity_figure[title_fig_sparsity_figures.format(data, nb_cluster, sparsity_value, init)].add_trace(go.Scatter(
            #                     x= x_axis_vals + x_axis_vals[::-1],
            #                     y=std_objectives_upper + std_objectives_lower[::-1],
            #                     fill='toself',
            #                     showlegend=False,
            #                     fillcolor='rgba{}'.format(color_by_n_iter[int(nb_iter)] + tpl_transparency),
            #                     line_color='rgba(255,255,255,0)',
            #                 name='Kmeans + PALM {}'.format(nb_iter)
            #                 ))

                           #  ###########################
                           #  # FIGURE NITER PALM FIXED #
                           #  ###########################
                           #  dct_niter_figure[title_fig_niter_figures.format(data, nb_cluster, nb_iter, init)].add_trace(go.Scatter(x=np.arange(len(mean_objectives)),
                           #                                                                                                         y=mean_objectives,
                           #                                                                                                         # error_y=dict(
                           #                                                                                                         #     type='data',  # value of error bar given in data coordinates
                           #                                                                                                         #     array=std_objectives,
                           #                                                                                                         #     visible=True
                           #                                                                                                         # ),
                           #                                                                                                         line=dict(color="rgb{}".format(color_by_sparsity[int(sparsity_value)]), dash="dot"),
                           #                                                                                                         mode='lines',
                           #                                                                                                         name='Kmeans + PALM {}'.format(sparsity_value)))
                           #
                           #  dct_niter_figure[title_fig_niter_figures.format(data, nb_cluster, nb_iter, init)].add_trace(go.Scatter(
                           #      x=x_axis_vals + x_axis_vals[::-1],
                           #      y=std_objectives_upper + std_objectives_lower[::-1],
                           #      fill='toself',
                           #      showlegend=False,
                           #      fillcolor='rgba{}'.format(color_by_sparsity[int(sparsity_value)] + tpl_transparency),
                           #      line_color='rgba(255,255,255,0)',
                           # name='Kmeans + PALM {}'.format(sparsity_value)
                           #  ))

                            # ##############################
                            # # FIGURE VARYING INIT METHOD #
                            # ##############################
                            # dct_init_figure[title_fig_init_figures.format(data, nb_cluster, str(sparsity_value), str(nb_iter))].add_trace(go.Scatter(x=np.arange(len(mean_objectives)),
                            #                                                                                                                          y=mean_objectives,
                            #                                                                                                                          # error_y=dict(
                            #                                                                                                                          #     type='data',
                            #                                                                                                                          #     # value of error bar given in data coordinates
                            #                                                                                                                          #     array=std_objectives,
                            #                                                                                                                          #     visible=True
                            #                                                                                                                          # ),
                            #                                                                                                                          line=dict(color="rgb{}".format(color_by_init[init]), dash="dot"),
                            #                                                                                                                          mode='lines',
                            #                                                                                                                          name='Kmeans + PALM {}'.format(init)))
                            # dct_init_figure[title_fig_init_figures.format(data, nb_cluster, str(sparsity_value), str(nb_iter))].add_trace(go.Scatter(
                            #     x=x_axis_vals + x_axis_vals[::-1],
                            #     y=std_objectives_upper + std_objectives_lower[::-1],
                            #     fill='toself',
                            #     showlegend=False,
                            #     fillcolor='rgba{}'.format(color_by_init[init] + tpl_transparency),
                            #     line_color='rgba(255,255,255,0)',
                            #  name='Kmeans + PALM {}'.format(dct_str_init[init])
                            # ))

            dct_outdir_fig = {
                "{}/sparsity_effect".format(data): dct_niter_figure,
                "{}/n_iter_effect".format(data): dct_sparsity_figure,
                "{}/init_effect".format(data): dct_init_figure
            }
            for outdir, dct_figs in dct_outdir_fig.items():
                outdir_figs = output_dir / outdir
                outdir_figs.mkdir(parents=True, exist_ok=True)
                for title, fig in dct_figs.items():
                    fig.update_layout(barmode='group',
                                      # title=title,
                                      autosize=False,
                                      width=500,
                                      height=500,
                                      xaxis_title="# Iteration",
                                      yaxis_title="Loss value",
                                      margin=dict(l=20, r=20, t=20, b=20),
                                      font=dict(
                                          # family="Courier New, monospace",
                                          size=18,
                                          color="black"
                                      ),
                                      legend=dict(
                                          # x=0, y=-0.3,
                                          traceorder="normal",
                                          font=dict(
                                              family="sans-serif",
                                              size=18,
                                              color="black"
                                          ),
                                          # bgcolor="LightSteelBlue",
                                          # bordercolor="Black",
                                          borderwidth=1,
                                      ),
                                      showlegend=False,
                                      )
                    fig.update_yaxes(type="log")
                    # fig.show()
                    title_file = title.replace(";", "_").replace(":", "_").replace(" ", "_").lower()
                    fig.write_image(str((outdir_figs / title_file).absolute()) + ".png")
