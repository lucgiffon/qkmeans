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
from decimal import Decimal

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
    2: (0, 128, 0), # green
    3: (0, 0, 255), # blue
    5: (0, 153, 153) # turquoise
}
color_by_n_iter = {
    50: (0, 128, 0), # green
    100: (0, 0, 255), # blue
    200: (102, 153, 255), # lightblue
    300: (0, 153, 153) # turquoise
}
color_by_init = {
    "kmeans++": (0, 128, 0), # green
    "uniform_sampling": (255, 0, 0) # ref
}

tpl_transparency = (0.2,)

dct_str_init = {
    "kmeans++": "Kmeans ++",
    "uniform_sampling": "Uniform"
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


    figures_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/reports/figures/")
    output_dir = figures_dir / suf_path / "curves"



    nb_iter_palm = set(df_results["nb-iteration-palm"].values)

    # hierarchical_values = set(df_results["hierarchical-init"])
    nb_iter_kmeans = max(set(df_results["nb-iteration"]))
    i = 0

    df_results = df_results[df_results["initialization"] == "kmeans++"]
    df_results_kmeans = df_results[df_results["model"] == "Kmeans"]
    # df_results = df_results[df_results["dataset"] == "Coverage Type"]
    df_results = df_results[df_results["nb-iteration-palm"] == 300]

    # df_results = df_results[df_results["sparsity-factor"] == 5]
    df_results = pd.concat([df_results, df_results_kmeans])
    df_results = df_results[df_results["dataset"] != "Kddcup04"]
    only_sparsity = 5
    df_results = df_results.loc[(df_results["sparsity-factor"] == "None") | (df_results["sparsity-factor"] == only_sparsity)]
    df_histo = df_results.loc[~(df_results["dataset"]== "Coverage Type") | ((df_results["dataset"]== "Coverage Type") & (df_results["nb-cluster"] == 256))]

    dct_results_table = dict()
    datasets = set(df_results["dataset"].values)
    for data in datasets:
        dct_results_table[data] = dict()
        df_data = df_results[df_results["dataset"] == data]
        nb_clusters = sorted(set(df_data["nb-cluster"].values))
        df_nb_cluster = df_data[df_data["nb-cluster"] == max(nb_clusters)]


        ###########
        # QKMEANS #
        ###########
        dct_results_table[data]["QKmeans"] = dict()
        df_qkmeans = df_nb_cluster[df_nb_cluster["model"] == "QKmeans"]
        for hierarchical_value in [True]:
            df_hierarchical = df_qkmeans[df_qkmeans["hierarchical-init"] == hierarchical_value]
            sparsity_values = set(df_results["sparsity-factor"].values)
            # sparsity_values.remove("None")
            for sparsity_value in sparsity_values:
                df_sparsity = df_hierarchical[df_hierarchical["sparsity-factor"] == sparsity_value]

                for nb_iter in nb_iter_palm:
                    df_iter = df_sparsity[df_sparsity["nb-iteration-palm"] == nb_iter]

                    init_schemes = set(df_iter["initialization"].values)
                    for init in init_schemes:
                        df_init = df_iter[df_iter["initialization"] == init]
                        objectives = [np.load(path, allow_pickle=True)["qmeans_objective"][1] for path in df_init["path-objective"]]
                        objectives = fill_objective_values(objectives, nb_iter_kmeans)  # remplis les objectifs pour qu'ils fassent tous la même taille

                        mean_objectives = np.mean(objectives, axis=0)
                        std_objectives = np.std(objectives, axis=0)

                        dct_results_table[data]["QKmeans"][sparsity_value] = (mean_objectives[-1], std_objectives[-1])


        ###########
        # KMEANS #
        ###########
        dct_results_table[data]["Kmeans"] = dict()
        df_kmeans = df_nb_cluster[df_nb_cluster["model"] == "Kmeans"]
        init_schemes = set(df_kmeans["initialization"].values)
        for init in init_schemes:
            df_init = df_kmeans[df_kmeans["initialization"] == init]
            objectives = [np.load(path, allow_pickle=True)["kmeans_objective"][1] for path in df_init["path-objective"]]
            objectives = fill_objective_values(objectives, nb_iter_kmeans)  # remplis les objectifs avec leur meilleur valeur pour qu'ils fassent tous la même taille
            mean_objectives = np.mean(objectives, axis=0)
            std_objectives = np.std(objectives, axis=0)

            dct_results_table[data]["Kmeans"] = (mean_objectives[-1], std_objectives[-1])



        ##################
        # KMEANS  + PALM #
        ##################
        dct_results_table[data]["Kmeans+Palm"] = dict()
        df_kmeans_palm = df_nb_cluster[df_nb_cluster["model"] == "Kmeans + Palm"]
        for hierarchical_value in [True]:
            df_hierarchical = df_kmeans_palm[df_kmeans_palm["hierarchical-inside"] == hierarchical_value]
            sparsity_values = set(df_hierarchical["sparsity-factor"].values)

            for sparsity_value in sparsity_values:
                df_sparsity = df_hierarchical[df_hierarchical["sparsity-factor"] == sparsity_value]
                nb_iter_palm = set(df_sparsity["nb-iteration-palm"].values)
                for nb_iter in nb_iter_palm:
                    df_iter = df_sparsity[df_sparsity["nb-iteration-palm"] == nb_iter]
                    init_schemes = set(df_iter["initialization"].values)
                    for init in init_schemes:
                        df_init = df_iter[df_iter["initialization"] == init]

                        objectives = [df_init["final-objective-value"] for _ in range(nb_iter_kmeans)]
                        # objectives = fill_objective_values(objectives, nb_iter_kmeans)  # remplis les objectifs pour qu'ils fassent tous la même taille

                        mean_objectives = np.mean(objectives, axis=1)
                        std_objectives = np.std(objectives, axis=1)

                        # name_method = f"Kmeans + PALM; K={nb_cluster}; e={sparsity_value}; {init}"
                        dct_results_table[data]["Kmeans+Palm"][sparsity_value] = (mean_objectives[-1], std_objectives[-1])

    table_results_mean = np.empty((len(dct_results_table) + 1, 1 + 1 + 2), dtype=object)
    table_results_std = np.empty_like(table_results_mean, dtype=object)
    table_results_mean[0][0] = "Dataset"
    order_sparsity_values = [only_sparsity]
    for idx_data, (data, dct_methods) in enumerate(dct_results_table.items()):
        table_results_mean[idx_data+1][0] = data
        table_results_std[idx_data+1][0] = data
        for name_method, dct_sparsity_or_tpl in dct_methods.items():
            if name_method == "Kmeans":
                table_results_mean[0][1] = "\kmeans"
                table_results_std[0][1] = "\kmeans"
                kmeans_mean, kmeans_std = dct_sparsity_or_tpl
                table_results_mean[idx_data+1][1] = kmeans_mean
                table_results_std[idx_data+1][1] = kmeans_std
            else:
                if name_method == "Kmeans+Palm":
                    start_idx = 3
                    name_method = "\kmeans~+~\palm"
                else:
                    start_idx = 2
                    name_method = "\qkmeans"
                for sparsity, tpl_results in dct_sparsity_or_tpl.items():
                    idx_sparsity = order_sparsity_values.index(sparsity)
                    table_results_mean[0][start_idx + idx_sparsity] = f"{name_method}"
                    table_results_std[0][start_idx + idx_sparsity] = f"{name_method}"

                    obj_mean, obj_std = tpl_results

                    table_results_mean[idx_data+1][start_idx + idx_sparsity] = obj_mean
                    table_results_std[idx_data+1][start_idx + idx_sparsity] = obj_std

    print(1)

    str_tabular = "\\toprule \n"
    # str_tabular += "&".join([table_results_mean[0][0]] + [f"\\texttt{{{elm}}}" for elm in table_results_mean[0][1:]])
    str_tabular += "&".join(table_results_mean[0])
    str_tabular += "\\\\ \n"
    str_tabular += "\\midrule \n"

    for idx_line, line in enumerate(table_results_mean[1:]):
        line_values = line[1:].astype(float)
        argsort_line_value = np.argsort(line_values)
        best_idx, snd_best_idx = argsort_line_value[0], argsort_line_value[1]
        reforged_values = []
        for idx_val, val in enumerate(line_values):
            if idx_val == best_idx:
                reforged_values.append('\\textbf{%.3E}'%Decimal(val))
            elif idx_val == snd_best_idx:
                reforged_values.append('\\underline{%.3E}'%Decimal(val))
            else:
                reforged_values.append('%.3E'%Decimal(val))

        reforged_values = ["\\texttt{{{elm}}}".format(elm=line[0])] + reforged_values
        str_line = "&".join(reforged_values)
        str_line += "\\\\ \n"
        str_tabular += str_line

    str_tabular += "\\bottomrule \n"
    print(str_tabular)

    r"""
    \begin{tabular}{@{}ccccccccc}
    \toprule                                                                                                                                                                              
                                                                &  Algorithm                  
                                                                %& \thead{\texttt{Blobs} \\ D=2000 \\ K=512}       
                                                                & \thead{\texttt{MNIST} \\ D=784 \\ K=64}   
                                                                & \thead{\texttt{F.-MNIST} \\ D=784 \\ K=64}     
                                                                & \thead{\texttt{B.-cancer} \\ D=30 \\ K=64}   
                                                                & \thead{\texttt{Covtype} \\ D=54 \\ K=512}     
                                                                & \thead{\texttt{Kddcup04} \\ D=74 \\ K=512}   
                                                                & \thead{\texttt{Kddcup99} \\ D=116 \\ K=512}
                                                                & \thead{\texttt{Caltech} \\ D=2352 \\ K=256}       \\ 
                                                
\midrule 
%                                                                                    & Mnist                          & Fashion Mnist               & Breast cancer                & Covtype              & Kddcup04              & Kddcup99           & Caltech
                                                                                                                                                                                                                                                                                     
\multirow{2}{*}{\shortstack{\# FLOP}}                           & \kmeans            & $100~352$                        & $100~352$                      & $3~840$                 & $55~296$                & $75~776$               & $118~784$              & $1~572~864$             \\
                                                                & \qkmeans           & \boldsymbol{$9~583$}             & \boldsymbol{$9~230$}           & \boldsymbol{$1~874$}    & \boldsymbol{$8~555$}     & \boldsymbol{$9~598$}   & \boldsymbol{$12~510$}   & \boldsymbol{$48~601$} \\
                          
\midrule              
                                                                                                                                                                                                                                                        
\multirow{2}{*}{\shortstack{\# non-zero \\ values}}             & \kmeans            & $50~176$                         & $50~176$                     & $1~920$                   & $27~648$                  & $37~888$               & $59~392$              & $786~432$  \\
                                                                & \qkmeans           & \boldsymbol{$4~791$}             & \boldsymbol{$4~615$}         & \boldsymbol{$937$}        & \boldsymbol{$4~277$}      & \boldsymbol{$4~799$}   & \boldsymbol{$6~255$}  & \boldsymbol{$24~300$}  \\

\midrule
                                                                                                                                                                                                                                                                             
\multirow{2}{*}{\shortstack{Compression \\ rate}}               & \multirow{2}{*}{\shortstack{$\ddfrac{\kmeans}{\qkmeans}$}}

                                                                & \multirow{2}{*}{$\times 10.4$}              % mnist  
                                                                & \multirow{2}{*}{$\times 10.9$}              % fashion mnist
                                                                & \multirow{2}{*}{$\times 2.0$}              % breast cancer
                                                                & \multirow{2}{*}{$\times 6.5$}              % covtype
                                                                & \multirow{2}{*}{$\times 7.9$}              % kddcup04
                                                                & \multirow{2}{*}{$\times 9.5$}            % kddcup99
                                                       			& \multirow{2}{*}{$\times 32.3$} \\            % caltech


&&&&&&\\


\midrule \midrule


\multirow{3}{*}{\shortstack{1-NN \\ Accuracy}}                  & \kmeans             & \underline{$0.9523$}           & \underline{$0.8339$}      & $0.924$                  & $0.9662$                & \underline{$0.6246$}    & \boldsymbol{$0.9990$}   & \boldsymbol{$0.1073$}      \\
                                                                & \qkmeans            & \underline{$0.9523$}           & $0.8367$                  & \boldsymbol{$0.93$}      & $0.9665$                & \boldsymbol{$0.6445$}   & \boldsymbol{$0.9990$}   & $0.0973$                   \\
                                                                & Ball-tree           & \boldsymbol{$0.9690$}          & \boldsymbol{$0.8497$}     & \underline{$0.9280$}     & $N/A$                   & $N/A$                   & $N/A$                   & $N/A$                      \\

  
\midrule \midrule                                                                                                                                                                                                                                                                                                
                                                        
                                                        

\multirow{5}{*}{\shortstack{Nyström \\ approximation \\ error}} & \kmeans            & \boldsymbol{$0.0322$}            & \boldsymbol{$0.0194$}       & \boldsymbol{$0.0022$}   & \underline{$0.0001$}     & \boldsymbol{$0.0011$}  & \boldsymbol{$0.0010$}  & \boldsymbol{$0.0138$}    \\
                                                                & \qkmeans           & \underline{$0.0454$}             & \underline{$0.0327$}        & \underline{$0.0024$}    & \boldsymbol{$1e^{-5}$}   & \underline{$0.0014$}   & \underline{$0.0021$}   & $0.026$                  \\
                                                                & Uniform            & $0.0673$                         & $0.0443$                    & $0.005$                 & \underline{$0.0001$}     & $0.0016$               & $0.0025$               & \underline{$0.0194$}     \\
                                                                & Un. F-Nys.         & $0.1702$                         & $0.2919$                    & $0.0449$                & $0.021$                  & $0.0213$               & $0.104$               & $0.2191$                 \\
                                                                & K. F-Nys.          & $0.1576$                         & $0.2623$                    & $0.0598$                & $0.0147$                 & $0.019$                & $0.1047$               & $0.2382$                 \\
 
\midrule 
 
  
\multirow{5}{*}{\shortstack{Nyström\\+\\SVM \\ Accuracy}}       & \kmeans              & \boldsymbol{$0.9244$}          & \boldsymbol{$0.8193$}       & \boldsymbol{$0.9340$}   & \underline{$0.6858$}     & $0.2611$                & \boldsymbol{$0.9989$}  & \boldsymbol{$0.1673$}   \\
                                                                & \qkmeans             & \underline{$0.9223$}           & $0.8111$                    & \boldsymbol{$0.9340$}   & $0.6854$                 & \underline{$0.2614$}    & $0.9977$               & $0.1566$                \\   
                                                                & Uniform              & $0.905$                        & \underline{$0.8142$}        & \boldsymbol{$0.9340$}   & \boldsymbol{$0.6868$}    & \boldsymbol{$0.2849$}   & \underline{$0.9982$}   & \underline{$0.1575$}    \\
                                                                & Un. F-Nys.           & $0.7937$                       & $0.7341$                    & $0.932$                 & $0.6552$                 & $0.2649$                & $0.995$                & $0.0954$                \\
                                                                & K. F-Nys.            & $0.7337$                       & $0.6872$                    & $0.93$                  & $0.6454$                 & $0.2609$                & $0.9956$               & $0.0751$                \\



\bottomrule
\end{tabular}}"""