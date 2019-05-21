"""
Analysis of objective function during qmeans execution

Usage:
  qmeans_objective_function_analysis kmeans [-h] [-v] [--output-file=str] [--seed=int] (--blobs) --nb-cluster=int --initialization=str [--nb-iteration=int]
  qmeans_objective_function_analysis qmeans [-h] [-v] [--output-file=str] [--seed=int] (--blobs) --nb-cluster=int --initialization=str --nb-factors=int --sparsity-factor=int [--hierarchical] [--nb-iteration-palm=int]

Options:
  -h --help                             Show this screen.
  -v --verbose                          Set verbosity to debug.
  --output-file=str                     Tell if the results should be written to some file and give the path to the file.
                                        The file name must be given without ext.
  --seed=int                            The seed to use for numpy random module.

Dataset:
  --blobs                               Use blobs dataset from sklearn.

Non-specific options:
  --nb-cluster=int                      Number of cluster to look for.
  --nb-iteration=int                    Number of iterations in the main algorithm. [default: 20]
  --initialization=str                  Desired type of initialization ('random', 'uniform_sampling'). For Qmeans, the initialized
                                        centroids are approximated by some sparse factors first using the HierarchicalPalm4msa algorithm..

Qmeans-Specifc options:
  --nb-factors=int                      Number of factors in the decomposition (without the extra upfront diagonal matrix).
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-iteration-palm=int               Number of iterations in the inner palm4msa calls. [default: 100]
  --residual-on-right                   Tells if the residual should be computed as right factor in each loop of hierarchical palm4msa.
  --update-right-to-left                Tells if the factors should be updated from right to left in each iteration of palm4msa.
  --hierarchical                        Uses the Hierarchical version of palm4msa inside the training loop

"""
from pathlib import Path

import docopt
import logging
import daiquiri
import time
import numpy as np
from pyqalm.utils import ResultPrinter, ParameterManager, ParameterManagerQmeans, ObjectiveFunctionPrinter
# todo graphical evaluation option
from pyqalm.qmeans import kmeans, qmeans, build_constraint_set_smart


def main_kmeans():
    X = paraman.get_dataset()
    U_init = paraman.get_initialization_centroids(X)
    start_kmeans = time.time()
    objective_values_k, final_centroids = kmeans(X_data=X,
           K_nb_cluster=paraman["--nb-cluster"],
           nb_iter=paraman["--nb-iteration"],
           initialization=U_init)
    stop_kmeans = time.time()
    kmeans_traintime = stop_kmeans - start_kmeans

    kmeans_results = {
        "traintime": kmeans_traintime
    }

    objprinter.add("kmeans_objective", ("after t"), objective_values_k)
    resprinter.add(kmeans_results)


def main_qmeans():
    X = paraman.get_dataset()
    U_init = paraman.get_initialization_centroids(X)

    lst_constraint_sets, lst_constraint_sets_desc = build_constraint_set_smart(left_dim=U_init.shape[0],
                                                                               right_dim=U_init.shape[1],
                                                                               nb_factors=paraman["--nb-factors"] + 1,
                                                                               sparsity_factor=paraman["--sparsity-factor"])

    parameters_palm4msa = {
        "init_lambda": 1.,
        "nb_iter": paraman["--nb-iteration-palm"],
        "lst_constraint_sets": lst_constraint_sets

    }

    start_qmeans = time.time()
    objective_values_q, centroid_factors, centroid_lambda = qmeans(X_data=X,
                                                                   K_nb_cluster=paraman["--nb-cluster"],
                                                                   nb_iter=paraman["--nb-iteration"],
                                                                   nb_factors=paraman["--nb-factors"] + 1,
                                                                   params_palm4msa=parameters_palm4msa,
                                                                   initialization=U_init,
                                                                   hierarchical_inside=paraman["--hierarchical"],
                                                                   )
    stop_qmeans = time.time()
    qmeans_traintime = stop_qmeans - start_qmeans
    qmeans_results = {
        "traintime": qmeans_traintime
    }


    objprinter.add("qmeans_objective", ("after palm", "after t"), objective_values_q)
    resprinter.add(qmeans_results)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManager(arguments)
    resprinter = ResultPrinter(output_file=paraman["--output-file_resprinter"])
    objprinter = ObjectiveFunctionPrinter(output_file=paraman["--output-file_objprinter"])

    if paraman["--verbose"]:
        daiquiri.setup(level=logging.DEBUG)
    else:
        daiquiri.setup(level=logging.INFO)

    if paraman["kmeans"]:
        main_kmeans()
    elif paraman["qmeans"]:
        paraman = ParameterManagerQmeans(paraman)
        main_qmeans()
        resprinter.add(paraman)


    else:
        raise NotImplementedError("Unknown method.")

    resprinter.print()
    objprinter.print()