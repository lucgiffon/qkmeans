"""
Analysis of objective function during qmeans execution

Usage:
  qmeans_objective_function_analysis kmeans [-h] [-v] [--output-file=str] [--seed=int] (--blobs|--census|--kddcup|--plants|--mnist|--fashion-mnist) --nb-cluster=int --initialization=str [--nb-iteration=int] [--assignation-time]
  qmeans_objective_function_analysis qmeans [-h] [-v] [--output-file=str] [--seed=int] (--blobs|--census|--kddcup|--plants|--mnist|--fashion-mnist) --nb-cluster=int --initialization=str --nb-factors=int --sparsity-factor=int [--hierarchical] [--nb-iteration-palm=int] [--residual-on-right] [--assignation-time]

Options:
  -h --help                             Show this screen.
  -v --verbose                          Set verbosity to debug.
  --output-file=str                     Tell if the results should be written to some file and give the path to the file.
                                        The file name must be given without ext.
  --seed=int                            The seed to use for numpy random module.

Dataset:
  --blobs                               Use blobs dataset from sklearn. # todo blobs dataset with K > d and d < K
  --census                              Use census dataset. # todo add description for all those datasets
  --kddcup                              Use Kddcupbio dataset.
  --plants                              Use plants dataset.
  --mnist                               Use mnist dataset.
  --fashion-mnist                       Use fasion-mnist dataset. # todo add writer for centroids result

Tasks:
  --assignation-time                    Evaluate assignation time for a single points when clusters have been defined.

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
from pyqalm.qmeans import kmeans, qmeans, build_constraint_set_smart, get_distances

lst_results_header = [
    "traintime",
    "assignation_mean_time",
    "assignation_std_time"
]
def main_kmeans(X, U_init):
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

    return final_centroids


def main_qmeans(X, U_init):
    lst_constraint_sets, lst_constraint_sets_desc = build_constraint_set_smart(left_dim=U_init.shape[0],
                                                                               right_dim=U_init.shape[1],
                                                                               nb_factors=paraman["--nb-factors"] + 1,
                                                                               sparsity_factor=paraman["--sparsity-factor"],
                                                                               residual_on_right=paraman["--residual-on-right"])

    parameters_palm4msa = {
        "init_lambda": 1.,
        "nb_iter": paraman["--nb-iteration-palm"],
        "lst_constraint_sets": lst_constraint_sets,
        "residual_on_right": paraman["--residual-on-right"]
    }

    start_qmeans = time.time()
    objective_values_q, final_centroids = qmeans(X_data=X,
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

    return final_centroids

def make_assignation_evaluation(X, centroids):
    nb_eval = 100
    times = []
    for i in np.random.permutation(X.shape[0])[:nb_eval]:
        start_time = time.time()
        get_distances(X[i].reshape(1, -1), centroids)
        stop_time = time.time()
        times.append(stop_time - start_time)

    mean_time = np.mean(times)
    std_time = np.std(times)

    resprinter.add({
        "assignation_mean_time": mean_time,
        "assignation_std_time": std_time
    })


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManager(arguments)
    initialized_results = dict((v, None) for v in lst_results_header)
    resprinter = ResultPrinter(initialized_results, output_file=paraman["--output-file_resprinter"])
    resprinter.add(paraman)
    objprinter = ObjectiveFunctionPrinter(output_file=paraman["--output-file_objprinter"])

    if paraman["--verbose"]:
        daiquiri.setup(level=logging.DEBUG)
    else:
        daiquiri.setup(level=logging.INFO)

    X = paraman.get_dataset()
    U_init = paraman.get_initialization_centroids(X)

    if paraman["kmeans"]:
        U_final = main_kmeans(X, U_init)
    elif paraman["qmeans"]:
        paraman_q = ParameterManagerQmeans(arguments)
        paraman.update(paraman_q)
        U_final = main_qmeans(X, U_init)
    else:
        raise NotImplementedError("Unknown method.")


    if paraman["--assignation-time"]:
        make_assignation_evaluation(X, U_final)


    resprinter.print()
    objprinter.print()