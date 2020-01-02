"""
Analysis of objective function during qmeans execution. This script is derived from `code/scripts/2019/07/qmeans_objective_function_analysis_better_timing.py`
and change in nystrom evaluation that is now normalized.

Usage:
  efficient_nystrom [-h] [-v|-vv] [--seed=int] (--blobs str|--light-blobs|--covtype|--breast-cancer|--census|--kddcup04|--kddcup99|--plants|--mnist|--fashion-mnist|--lfw|--caltech256 int|--million-blobs int) --nb-landmarks=int [--max-eval-train-size=init] --nystrom=int

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --seed=int                            The seed to use for numpy random module.

Dataset:
  --blobs str                           Use blobs dataset from sklearn. Formatting is size-dimension-nbcluster
  --light-blobs                         Use blobs dataset from sklearn with few data for testing purposes.
  --census                              Use census dataset.
  --kddcup04                            Use Kddcupbio dataset.
  --kddcup99                            Use 10 percent of Kddcup99 dataset.
  --plants                              Use plants dataset.
  --mnist                               Use mnist dataset.
  --fashion-mnist                       Use fasion-mnist dataset.
  --lfw                                 Use Labeled Faces in the Wild dataset.
  --caltech256 int                      Use caltech256 dataset with square images of size int.
  --million-blobs int                   Use the million blobs dataset with int million.
  --breast-cancer                       Use breast cancer dataset from sklearn.
  --covtype                             Use breast cancer dataset from sklearn.

Non-specific options:
  --nb-landmarks=int                    Number of landmark points in the final approximation.
  --max-eval-train-size=int             Max size of train for evaluation
  --nystrom=int                         Size of evaluation set for nystrom
"""
import signal
import docopt
import logging
import daiquiri
import sys
import time
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from qkmeans.FWHT import FWHT
from qkmeans.data_structures import SparseFactors, UfastOperator
from qkmeans.kernel.kernel import special_rbf_kernel, nystrom_transformation, prepare_nystrom
from qkmeans.palm.palm_fast import hierarchical_palm4msa, palm4msa
from qkmeans.core.kmeans_minibatch import kmeans_minibatch
from qkmeans.core.qmeans_minibatch import qkmeans_minibatch
from qkmeans.utils import ResultPrinter, ParameterManager, ObjectiveFunctionPrinter, logger, timeout_signal_handler, compute_euristic_gamma, log_memory_usage, next_power_of_two, \
    ParameterManagerEfficientNystrom
from qkmeans.core.qmeans_fast import qmeans, init_lst_factors
from qkmeans.core.utils import build_constraint_set_smart, get_distances, get_squared_froebenius_norm_line_wise
from qkmeans.core.kmeans import kmeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse.linalg import LinearOperator
from sklearn.svm import LinearSVC

lst_results_header = [
    "nystrom_sampled_error_reconstruction_uop",
    "nystrom_svm_accuracy_uop",
    "nystrom_sampled_error_reconstruction_uniform",
    "nystrom_svm_accuracy_uniform",
    "nystrom_sampled_error_reconstruction_kmeans",
    "nystrom_svm_accuracy_kmeans",
    "nystrom_sampled_error_reconstruction_uop_kmeans",
    "nystrom_svm_accuracy_uop_kmeans",
    "nystrom_sampled_error_reconstruction_seeds",
    "nystrom_svm_accuracy_seeds"
]

def make_nystrom_evaluation(x_train, y_train, x_test, y_test, gamma, landmarks):
    # verify sample size for evaluation
    n_sample = paraman["--nystrom"]
    if n_sample > x_train.shape[0]:
        logger.warning("Batch size for nystrom evaluation is bigger than data size. {} > {}. Using "
                       "data size instead.".format(n_sample, x_train.shape[0]))
        n_sample = x_train.shape[0]
        paraman["--nystrom"] = n_sample

    indexes_samples = np.random.permutation(x_train.shape[0])[:n_sample]
    sample = x_train[indexes_samples]
    samples_norm = None

    # Make nystrom approximation
    landmarks_norm = get_squared_froebenius_norm_line_wise(landmarks)[:, np.newaxis]
    metric = prepare_nystrom(landmarks, landmarks_norm, gamma=gamma)
    nystrom_embedding = nystrom_transformation(sample, landmarks, metric, landmarks_norm, samples_norm, gamma=gamma)
    nystrom_approx_kernel_value = nystrom_embedding @ nystrom_embedding.T

    # Create real kernel matrix
    real_kernel_special = special_rbf_kernel(sample, sample, gamma, norm_X=samples_norm, norm_Y=samples_norm)
    # real_kernel = rbf_kernel(sample, sample, gamma)
    real_kernel_norm = np.linalg.norm(real_kernel_special)

    # evaluation reconstruction error
    reconstruction_error_nystrom = np.linalg.norm(nystrom_approx_kernel_value - real_kernel_special) / real_kernel_norm

    # start svm + nystrom classification
    if x_test is not None:
        logger.info("Start classification")

        x_train_nystrom_embedding = nystrom_transformation(x_train, landmarks, metric, landmarks_norm, None, gamma=gamma)
        x_test_nystrom_embedding = nystrom_transformation(x_test, landmarks, metric, landmarks_norm, None, gamma=gamma)

        linear_svc_clf = LinearSVC(class_weight="balanced")
        linear_svc_clf.fit(x_train_nystrom_embedding, y_train)
        predictions = linear_svc_clf.predict(x_test_nystrom_embedding)

        if paraman["--kddcup04"]:
            # compute recall: nb_true_positive/real_nb_positive
            recall = np.sum(predictions[y_test == 1])/np.sum(y_test[y_test == 1])
            # compute precision: nb_true_positive/nb_positive
            precision = np.sum(predictions[y_test == 1])/np.sum(predictions[predictions==1])
            f1 = 2 * precision * recall / (precision + recall)
            accuracy_nystrom_svm = f1
        else:
            accuracy_nystrom_svm = np.sum(predictions == y_test) / y_test.shape[0]

    else:
        accuracy_nystrom_svm = None

    return reconstruction_error_nystrom, accuracy_nystrom_svm


def main_kmeans(X, U_init):
    """
    Will perform the k means algorithm on X with U_init as initialization

    :param X: The input data in which to find the clusters.
    :param U_init: The initialization of the the clusters.

    :return: The final centroids, the indicator vector
    """
    if paraman["--minibatch"]:
        objective_values_k, final_centroids, indicator_vector_final = kmeans_minibatch(X_data=X,
                                                                                       K_nb_cluster=U_init.shape[0],
                                                                                       nb_iter=paraman["--nb-iteration"],
                                                                                       initialization=U_init,
                                                                                       batch_size=paraman["--minibatch"]
                                                                                       )
    else:
        objective_values_k, final_centroids, indicator_vector_final = kmeans(X_data=X,
               K_nb_cluster=U_init.shape[0],
               nb_iter=paraman["--nb-iteration"],
               initialization=U_init)

    return final_centroids, indicator_vector_final

if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerEfficientNystrom(arguments)
    initialized_results = dict((v, None) for v in lst_results_header)
    resprinter = ResultPrinter(output_file=paraman["--output-file_resprinter"])
    resprinter.add(initialized_results)
    resprinter.add(paraman)
    has_failed = False
    if paraman["-v"] >= 2:
        daiquiri.setup(level=logging.DEBUG)
    elif paraman["-v"] >= 1:
        daiquiri.setup(level=logging.INFO)
    else:
        daiquiri.setup(level=logging.WARNING)

    logging.warning("Verbosity set to warning")
    logging.info("Verbosity set to info")
    logging.debug("Verbosity set to debug")


    try:
        dataset = paraman.get_dataset()

        if dataset["x_train"].dtype != np.float32:
            dataset["x_train"] = dataset["x_train"].astype(np.float32)


        sizes = {
            "size_train": dataset["x_train"].shape[0]
        }

        if "x_test" in dataset:
            if dataset["x_train"].dtype != np.float32:
                dataset["x_test"] = dataset["x_test"].astype(np.float32)
                dataset["y_test"] = dataset["y_test"].astype(np.float32)
                dataset["y_train"] = dataset["y_train"].astype(np.float32)
            sizes.update({
                "size_test": dataset["x_test"].shape[0]
            })

        resprinter.add(sizes)

        log_memory_usage("Memory after loading dataset and initialization of centroids")

        if paraman["--max-eval-train-size"] is not None:
            train_indexes = np.random.choice(dataset["x_train"].shape[0], size=paraman["--max-eval-train-size"], replace=False)
        else:
            train_indexes = np.arange(dataset["x_train"].shape[0])

        logger.info("Train size: {}".format(train_indexes.size))


        scaler = StandardScaler(with_std=False)
        data_train = dataset["x_train"][train_indexes]
        data_train = scaler.fit_transform(data_train)
        deficit_dim_before_power_of_two = 2**next_power_of_two(data_train.shape[1]) - data_train.shape[1]
        data_train = np.pad(data_train, [(0, 0), (0, deficit_dim_before_power_of_two)], 'constant')

        gamma = compute_euristic_gamma(data_train)

        logger.info("Start Nyström reconstruction evaluation".format(paraman["--nystrom"]))
        if "x_test" in dataset.keys():
            data_test = dataset["x_test"]
            data_test = scaler.transform(data_test)
            data_test = np.pad(data_test, [(0, 0), (0, deficit_dim_before_power_of_two)], 'constant')

            nystrom_eval = lambda landmarks: make_nystrom_evaluation(x_train=data_train,
                                                                     y_train=dataset["y_train"][train_indexes],
                                                                     x_test=data_test,
                                                                     y_test=dataset["y_test"],
                                                                     gamma=gamma,
                                                                     landmarks=landmarks)
        else:
            nystrom_eval = lambda landmarks: make_nystrom_evaluation(x_train=data_train,
                                                                     y_train=None,
                                                                     x_test=None,
                                                                     y_test=None,
                                                                     gamma=gamma,
                                                                     landmarks=landmarks)

        nb_seeds = (paraman["--nb-landmarks"] // data_train.shape[1]) + 1
        seeds = data_train[np.random.permutation(data_train.shape[0])[:nb_seeds]]
        uop = UfastOperator(seeds, FWHT)
        uop_arr = uop.toarray()

        if paraman["--nb-landmarks"] < uop_arr.shape[0]:
            uop_arr = uop_arr[:-(uop_arr.shape[0] - paraman["--nb-landmarks"]), :]

        uniform_sample = data_train[np.random.permutation(data_train.shape[0])[:uop_arr.shape[0]]]
        kmeans_sample, _ = main_kmeans(data_train, uniform_sample)

        seeds_kmeans = kmeans_sample[:nb_seeds]
        uop_kmeans = UfastOperator(seeds_kmeans, FWHT)
        uop_arr_kmeans = uop_kmeans.toarray()

        if paraman["--nb-landmarks"] < uop_arr_kmeans.shape[0]:
            uop_arr_kmeans = uop_arr_kmeans[:-(uop_arr_kmeans.shape[0] - paraman["--nb-landmarks"]), :]

        reconstruction_error_nystrom_uop, accuracy_nystrom_svm_uop = nystrom_eval(uop_arr)
        reconstruction_error_nystrom_seeds, accuracy_nystrom_svm_seeds = nystrom_eval(seeds)
        reconstruction_error_nystrom_uniform, accuracy_nystrom_svm_uniform = nystrom_eval(uniform_sample)
        reconstruction_error_nystrom_kmeans, accuracy_nystrom_svm_kmeans = nystrom_eval(kmeans_sample)
        reconstruction_error_nystrom_uop_kmeans, accuracy_nystrom_svm_uop_kmeans = nystrom_eval(uop_arr_kmeans)

        results = {
            "nb_seeds": nb_seeds,
            "nystrom_sampled_error_reconstruction_uop": reconstruction_error_nystrom_uop,
            "nystrom_svm_accuracy_uop":accuracy_nystrom_svm_uop,
            "nystrom_sampled_error_reconstruction_uniform":reconstruction_error_nystrom_uniform,
            "nystrom_svm_accuracy_uniform":accuracy_nystrom_svm_uniform,
            "nystrom_sampled_error_reconstruction_kmeans":reconstruction_error_nystrom_kmeans,
            "nystrom_svm_accuracy_kmeans":accuracy_nystrom_svm_kmeans,
            "nystrom_sampled_error_reconstruction_uop_kmeans":reconstruction_error_nystrom_uop_kmeans,
            "nystrom_svm_accuracy_uop_kmeans":accuracy_nystrom_svm_uop_kmeans,
            "nystrom_sampled_error_reconstruction_seeds":reconstruction_error_nystrom_seeds,
            "nystrom_svm_accuracy_seeds":accuracy_nystrom_svm_seeds
        }
        resprinter.add(results)

    except Exception as e:
        has_failed = True
        failure_dict = {
            "failure": has_failed
        }

        resprinter.add(failure_dict)
        resprinter.print()
        raise e

    failure_dict = {
        "failure": has_failed
    }

    resprinter.add(failure_dict)
    resprinter.print()
