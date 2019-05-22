"""
Analysis of objective function during qmeans execution

Usage:
  qmeans_objective_function_analysis kmeans [-h] [-v] --output-file=str [--seed=int] (--blobs|--census|--kddcup|--plants|--mnist|--fashion-mnist) --nb-cluster=int --initialization=str [--nb-iteration=int] [--assignation-time] [--1-nn]
  qmeans_objective_function_analysis qmeans [-h] [-v] --output-file=str [--seed=int] (--blobs|--census|--kddcup|--plants|--mnist|--fashion-mnist) --nb-cluster=int --initialization=str --nb-factors=int --sparsity-factor=int [--hierarchical] [--nb-iteration-palm=int] [--residual-on-right] [--assignation-time] [--1-nn]

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
  --1-nn                                Evaluate inference time (by instance) and inference accuracy for 1-nn (available only for mnist and fashion-mnist datasets)

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
import signal
import docopt
import logging
import daiquiri
import time
import numpy as np
from pyqalm.utils import ResultPrinter, ParameterManager, ParameterManagerQmeans, ObjectiveFunctionPrinter, logger, timeout_signal_handler
# todo graphical evaluation option
from pyqalm.qmeans import kmeans, qmeans, build_constraint_set_smart, get_distances
from sklearn.neighbors import KNeighborsClassifier

lst_results_header = [
    "traintime",
    "assignation_mean_time",
    "assignation_std_time",
    "1nn_kmean_inference_time",
    "1nn_kmean_accuracy",
    "1nn_brute_inference_time",
    "1nn_brute_accuracy",
    "1nn_kd_tree_inference_time",
    "1nn_kd_tree_accuracy",
    "1nn_ball_tree_inference_time",
    "1nn_ball_tree_accuracy"
]

def main_kmeans(X, U_init):
    start_kmeans = time.time()
    objective_values_k, final_centroids, indicator_vector_final = kmeans(X_data=X,
           K_nb_cluster=paraman["--nb-cluster"],
           nb_iter=paraman["--nb-iteration"],
           initialization=U_init)
    stop_kmeans = time.time()
    kmeans_traintime = stop_kmeans - start_kmeans

    kmeans_results = {
        "traintime": kmeans_traintime
    }

    objprinter.add("kmeans_objective", ("after t", ), objective_values_k)
    resprinter.add(kmeans_results)

    return final_centroids, indicator_vector_final


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
    objective_values_q, final_centroids, indicator_vector_final = qmeans(X_data=X,
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

    return final_centroids, indicator_vector_final

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


def make_1nn_evaluation(x_train, y_train, x_test, y_test, U_centroids, indicator_vector):

    def scikit_evaluation(str_type):
        clf = KNeighborsClassifier(n_neighbors=1, algorithm=str_type)
        clf.fit(x_train, y_train)

        start_inference_time = time.time()
        predictions = np.empty_like(y_test)
        for obs_idx, obs_test in enumerate(x_test):
            predictions[obs_idx] = clf.predict(obs_test.reshape(1, -1))[0]
        stop_inference_time = time.time()

        inference_time = (stop_inference_time - start_inference_time)

        accuracy = np.sum(predictions == y_test) / y_test.shape[0]

        results_1nn = {
            "1nn_{}_inference_time".format(str_type): inference_time,
            "1nn_{}_accuracy".format(str_type): accuracy
        }
        resprinter.add(results_1nn)
        return inference_time

    def kmean_tree_evaluation():

        lst_clf_by_cluster = [KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(x_train[indicator_vector == i], y_train[indicator_vector == i]) for i in range(U_centroids.shape[0])]

        start_inference_time = time.time()
        distances = get_distances(x_test, U_centroids)
        indicator_vector_test = np.argmin(distances, axis=1)
        predictions = np.empty_like(y_test)
        for obs_idx, obs_test in enumerate(x_test):
            idx_cluster = indicator_vector_test[obs_idx]
            clf_cluster = lst_clf_by_cluster[idx_cluster]
            predictions[obs_idx] = clf_cluster.predict(obs_test.reshape(1, -1))[0]

        stop_inference_time = time.time()
        inference_time = (stop_inference_time - start_inference_time)

        accuracy = np.sum(predictions == y_test) / y_test.shape[0]

        results_1nn = {
            "1nn_kmean_inference_time": inference_time,
            "1nn_kmean_accuracy": accuracy
        }
        resprinter.add(results_1nn)
        return inference_time

    logger.info("1 nearest neighbor with k-means search")
    kmean_tree_time = kmean_tree_evaluation()
    if paraman["kmeans"]:
        lst_knn_types = ["brute", "ball tree", "kd tree"]
        for knn_type in lst_knn_types:
            signal.signal(signal.SIGALRM, timeout_signal_handler)
            signal.alarm(int(kmean_tree_time * 10))
            try:
                logger.info("1 nearest neighbor with {} search".format(knn_type))
                scikit_evaluation(knn_type)
            except TimeoutError as te:
                logger.warning("Timeout during execution of 1-nn with {} version: {}".format(knn_type, te))
            signal.alarm(0)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManager(arguments)
    initialized_results = dict((v, None) for v in lst_results_header)
    resprinter = ResultPrinter(output_file=paraman["--output-file_resprinter"])
    resprinter.add(initialized_results)
    resprinter.add(paraman)
    objprinter = ObjectiveFunctionPrinter(output_file=paraman["--output-file_objprinter"])

    if paraman["--verbose"]:
        daiquiri.setup(level=logging.DEBUG)
    else:
        daiquiri.setup(level=logging.INFO)

    dataset = paraman.get_dataset()

    dataset["x_train"] = dataset["x_train"].astype(np.float)
    if "x_test" in dataset:
        dataset["x_test"] = dataset["x_test"].astype(np.float)
        dataset["y_test"] = dataset["y_test"].astype(np.float)
        dataset["y_train"] = dataset["y_train"].astype(np.float)

    U_init = paraman.get_initialization_centroids(dataset["x_train"])

    if paraman["kmeans"]:
        U_final, indicator_vector_final = main_kmeans(dataset["x_train"], U_init)
    elif paraman["qmeans"]:
        paraman_q = ParameterManagerQmeans(arguments)
        paraman.update(paraman_q)
        U_final, indicator_vector_final = main_qmeans(dataset["x_train"], U_init)
    else:
        raise NotImplementedError("Unknown method.")

    np.save(paraman["--output-file_centroidprinter"], U_final, allow_pickle=True)

    if paraman["--assignation-time"]:
        make_assignation_evaluation(dataset["x_train"], U_final)

    if paraman["--1-nn"]:
        logger.info("Start 1 nearest neighbor evaluation")
        make_1nn_evaluation(x_train=dataset["x_train"], # todo remove this
                            y_train=dataset["y_train"],
                            x_test=dataset["x_test"],
                            y_test=dataset["y_test"],
                            U_centroids=U_final,
                            indicator_vector=indicator_vector_final)


    resprinter.print()
    objprinter.print()