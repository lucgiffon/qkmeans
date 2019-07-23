import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
from io import StringIO
from pandas.errors import EmptyDataError
from pyqalm.utils import logger
from visualization.utils import get_dct_result_files_by_root, display_cmd_lines_from_root_name_list


src_results_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/results/2019/07/qmeans_analysis_blobs_reasonable")
# src_results_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/results/2019-05/qmeans_analysis_bigdataset_3_20_ghz_cpu_mem_gt_90")
if __name__ == "__main__":

    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir)

    dct_root_names = { 'root_names_linalgerror ': [],
                       'root_names_memoryerror': [],
                       'root_names_arpackerror': [],
                       'root_names_clusternopoint': [],
                       'root_names_jobkilled': [],
                       'root_names_unknown': [],
                       'root_names_sucess': [],
                       'root_names_nothing_job_killed': [],
                       'root_names_nothing': [],
                       'root_names_couldnotbebroadcast': [],
                       'root_names_failure': []}

    for root_name, dct_files in dct_output_files_by_root.items():
        stderr_file = src_results_dir / (root_name + ".stderr")
        with open(stderr_file, 'r') as stderr_file:
            end_of_err_file = "".join(stderr_file.readlines()[-50:])

        if len(dct_files) == 0:
            if "KILLED" in end_of_err_file:
                dct_root_names["root_names_nothing_job_killed"].append(root_name)
            else:
                dct_root_names["root_names_nothing"].append(root_name)

        if len(dct_files) != 3:
            if "numpy.linalg.LinAlgError" in end_of_err_file:
                dct_root_names["root_names_linalgerror"].append(root_name)
            elif "MemoryError" in end_of_err_file:
                dct_root_names["root_names_memoryerror"].append(root_name)
            elif "scipy.sparse.linalg.eigen.arpack.arpack.ArpackError" in end_of_err_file:
                # print(end_of_err_file)
                dct_root_names["root_names_arpackerror"].append(root_name)
            elif "Some clusters have no point" in end_of_err_file:
                dct_root_names["root_names_clusternopoint"].append(root_name)
                print(end_of_err_file)
            elif "KILLED" in end_of_err_file:
                dct_root_names["root_names_jobkilled"].append(root_name)
            elif "operands could not be broadcast together" in end_of_err_file:
                dct_root_names["root_names_couldnotbebroadcast"].append(root_name)
            else:
                print(end_of_err_file)
                dct_root_names["root_names_unknown"].append(root_name)
        else:
            result_file = src_results_dir / dct_files["results"]
            df = pd.read_csv(result_file)
            if df["failure"].all():
                if "Found array with 0 sample" in end_of_err_file:
                    dct_root_names["root_names_clusternopoint"].append(root_name)
                    print(end_of_err_file)
                else:
                    dct_root_names["root_names_failure"].append(root_name)
                    print(end_of_err_file)
            else:
                dct_root_names["root_names_sucess"].append(root_name)


    for execution_kind, lst_root_names_execution_kind in dct_root_names.items():
        print(execution_kind)
        print("Nb files {}:".format(execution_kind), len(lst_root_names_execution_kind))
        cmd_lines_execution_kind = display_cmd_lines_from_root_name_list(lst_root_names_execution_kind, src_results_dir, find_in_stderr=True)
        print("Nb cmd lines {}:".format(execution_kind), len(cmd_lines_execution_kind))
        for cmd in cmd_lines_execution_kind:
            print(cmd)
        print()
