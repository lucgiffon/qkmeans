import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
from io import StringIO
from pandas.errors import EmptyDataError

output_file_end_re = {
    "centroids": r"_centroids.npy",
    "results": r"_results.csv",
    "objective": r"_objective_.+.csv"
}

def display_cmd_lines_from_root_name_list(root_names_list):
    cmd_lines = []

    print("\n\n\n")
    for root_name in root_names_list:
        stdout_file = src_results_dir / (root_name + ".stdout")
        with open(stdout_file, 'r') as stdoutfile:
            lines = stdoutfile.readlines()
            i_line = -1
            for i_line, lin in enumerate(lines):
                if lin[:2] == "--":
                    break
            if i_line == -1:
                continue
            data = "".join(lines[i_line:i_line + 2])

            io_data = StringIO(data)
            df = pd.read_csv(io_data)

            # print("".join(lines))

        # print("\n")

        cmd_line = ""
        cmd_line += "qmeans" if df["qmeans"][0] else "kmeans"
        cmd_line += " --sparsity-factor" + " " + str(df["--sparsity-factor"][0])
        cmd_line += " --seed" + " " + str(df["--seed"][0])
        cmd_line += " --nystrom" if df["--nystrom"][0] else ""
        cmd_line += " --assignation-time" if df["--assignation-time"][0] else ""
        cmd_line += " --1-nn" if df["--1-nn"][0] else ""
        cmd_line += " --initialization" + " " + str(df["--initialization"][0])
        cmd_line += " --nb-cluster" + " " + str(df["--nb-cluster"][0])
        cmd_line += " --blobs" if df["--blobs"][0] else ""
        cmd_line += " --census" if df["--census"][0] else ""
        cmd_line += " --kddcup" if df["--kddcup"][0] else ""
        cmd_line += " --mnist" if df["--mnist"][0] else ""
        cmd_line += " --fashion-mnist" if df["--fashion-mnist"][0] else ""

        cmd_lines.append(cmd_line)

    for cmd_line in cmd_lines:
        print(cmd_line)
    print(len(cmd_lines))


def get_dct_

src_results_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/results/2019-05/big_expe_for_real")
if __name__ == "__main__":
    files =  src_results_dir.glob('**/*')
    files = [x for x in files if x.is_file()]
    lst_str_filenames = [file.name for file in files]

    dct_output_files_by_root = {}
    count_complete = 0
    count_has_printed_results = 0
    count_total = 0

    for pth_file in files:
        if pth_file.suffix != '.stdout':
            continue

        count_total += 1

        with open(pth_file, 'r') as stdoutfile:
            lines = stdoutfile.readlines()
            for i_line, lin in enumerate(lines):
                if lin[:2] == "--":
                    break
            else:
                print("file {} didn't contain anything".format(pth_file.name))
                dct_output_files_by_root[pth_file.stem] = {}
                continue
            count_has_printed_results += 1

            data = "".join(lines[i_line:i_line+2])

        io_data = StringIO(data)
        df = pd.read_csv(io_data)

        try:
            root_name = df["--output-file_resprinter"][0].split("_")[0]
        except KeyError:
            print("no key for resprinter in {}".format(pth_file.name))


        dct_files = {}
        complete = True
        for type_file, root_re in output_file_end_re.items():
            forged_re_compiled = re.compile(r"{}".format(root_name) + root_re)
            try:
                dct_files[type_file] = list(filter(forged_re_compiled.match, lst_str_filenames))[0]
            except IndexError:
                print("{} not found for root name {}".format(type_file, root_name))
                complete = False

        if complete:
            count_complete += 1


        dct_output_files_by_root[pth_file.stem] = dct_files

    print("Exploring values with no_results at all")

    for root_name, dct_files in dct_output_files_by_root.items():
        if len(dct_files) == 0:
            stderr_file = src_results_dir / (root_name + ".stderr")
            print(stderr_file)
            with open(stderr_file, 'r') as stderr_file:
                print(stderr_file.read())
            print()

    print("Exploring values with missing results")

    root_names_linalgerror = []
    root_names_memoryerror= []
    root_names_arpackerror = []
    root_names_clusternopoint = []
    root_names_jobkilled = []

    for root_name, dct_files in dct_output_files_by_root.items():
        if len(dct_files) != 3:
            stderr_file = src_results_dir / (root_name + ".stderr")
            print(stderr_file)
            with open(stderr_file, 'r') as stderr_file:
                end_of_err_file = "".join(stderr_file.readlines()[-10:])
                print(end_of_err_file)
            print()
            if "numpy.linalg.LinAlgError" in end_of_err_file:
                root_names_linalgerror.append(root_name)
            elif "MemoryError" in end_of_err_file:
                root_names_memoryerror.append(root_name)
            elif "scipy.sparse.linalg.eigen.arpack.arpack.ArpackError" in end_of_err_file:
                root_names_arpackerror.append(root_name)
            elif "Some clusters have no point" in end_of_err_file:
                root_names_clusternopoint.append(root_name)
            elif "KILLED ##" in end_of_err_file:
                root_names_jobkilled.append(root_name)

    print("Exploring successfull experiments")
    root_names_sucess = []
    for root_name, dct_files in dct_output_files_by_root.items():
        if len(dct_files) == 3:
            root_names_sucess.append(root_name)

    print()
    print("cmd lines linalgerror")
    print()
    display_cmd_lines_from_root_name_list(root_names_linalgerror)

    print()
    print("cmd lines memoryerror")
    print()
    display_cmd_lines_from_root_name_list(root_names_memoryerror)

    print()
    print("cmd lines arpackerror")
    print()
    display_cmd_lines_from_root_name_list(root_names_arpackerror)


    # print()
    # print("cmd lines cluster no poiint")
    # print()
    # display_cmd_lines_from_root_name_list(root_names_clusternopoint)

    print()
    print("cmd lines jobkilled")
    print()
    display_cmd_lines_from_root_name_list(root_names_jobkilled)

    print()
    print("cmd lines success")
    print()
    display_cmd_lines_from_root_name_list(root_names_sucess)


    val = dct_files