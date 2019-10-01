"""
Analysis of objective function during qmeans execution. This script is derived from `code/scripts/2019/07/qmeans_objective_function_analysis_better_timing.py`
and change in nystrom evaluation that is now normalized.

Usage:
  qmeans_objective_function_analysis kmeans [-h] [-v|-vv] [--seed=int] (--blobs str|--light-blobs|--census|--kddcup|--plants|--mnist|--fashion-mnist|--lfw|--caltech256 int) --nb-cluster=int --initialization=str [--nb-iteration=int] [--assignation-time=int] [--1-nn] [--nystrom=int] [--batch-assignation-time=int]
  qmeans_objective_function_analysis kmeans palm [-v|-vv] [-v] [--seed=int] (--blobs str|--light-blobs|--census|--kddcup|--plants|--mnist|--fashion-mnist|--lfw|--caltech256 int) --nb-cluster=int --initialization=str [--nb-iteration=int] [--assignation-time=int] [--1-nn] [--nystrom=int] [--batch-assignation-time=int] [--nb-iteration-palm=int] [--nb-factors=int] --sparsity-factor=int [--hierarchical]
  qmeans_objective_function_analysis qmeans [-h] [-v|-vv] [--seed=int] (--blobs str|--light-blobs|--census|--kddcup|--plants|--mnist|--fashion-mnist|--lfw|--caltech256 int) --nb-cluster=int --initialization=str [--nb-factors=int] --sparsity-factor=int [--hierarchical] [--nb-iteration=int] [--nb-iteration-palm=int] [--assignation-time=int] [--1-nn] [--nystrom=int] [--batch-assignation-time=int]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --seed=int                            The seed to use for numpy random module.

Dataset:
  --blobs str                           Use blobs dataset from sklearn. Formatting is size-dimension-nbcluster
  --light-blobs                         Use blobs dataset from sklearn with few data for testing purposes.
  --census                              Use census dataset.
  --kddcup                              Use Kddcupbio dataset.
  --plants                              Use plants dataset.
  --mnist                               Use mnist dataset.
  --fashion-mnist                       Use fasion-mnist dataset.
  --lfw                                 Use Labeled Faces in the Wild dataset.
  --caltech256 int                      Use caltech256 dataset with square images of size int.


Tasks:
  --assignation-time=int                Evaluate assignation time for a single points when clusters have been defined.
  --batch-assignation-time=int          Evaluate assignation time for a matrix of points when clusters have been defined. The integer is the number of data points to be considered.
  --1-nn                                Evaluate inference time (by instance) and inference accuracy for 1-nn (available only for mnist and fashion-mnist datasets)
  --nystrom=int                         Evaluate reconstruction time and reconstruction accuracy for Nystrom approximation. The integer is the number of sample for which to compute the nystrom transformation.

Non-specific options:
  --nb-cluster=int                      Number of cluster to look for.
  --nb-iteration=int                    Number of iterations in the main algorithm. [default: 20]
  --initialization=str                  Desired type of initialization ('random', 'uniform_sampling'). For Qmeans, the initialized
                                        centroids are approximated by some sparse factors first using the HierarchicalPalm4msa algorithm..

Palm-Specifc options:
  --nb-factors=int                      Number of factors in the decomposition (without the extra upfront diagonal matrix).
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-iteration-palm=int               Number of iterations in the inner palm4msa calls. [default: 300]
  --hierarchical                        Uses the Hierarchical version of palm4msa inside the training loop

"""
# todo add   --update-right-to-left                Tells if the factors should be updated from right to left in each iteration of palm4msa.
# todo --residual-on-right                   Tells if the residual should be computed as right factor in each loop of hierarchical palm4msa.
import signal
import docopt
import logging
import daiquiri
import sys
import time
import numpy as np
from qkmeans.data_structures import SparseFactors
from qkmeans.palm.palm_fast import hierarchical_palm4msa, palm4msa
from qkmeans.utils import ResultPrinter, ParameterManager, ObjectiveFunctionPrinter, logger, timeout_signal_handler, compute_euristic_gamma, log_memory_usage
# todo graphical evaluation option
from qkmeans.qk_means.qmeans_fast import qmeans, init_lst_factors
from qkmeans.qk_means.utils import build_constraint_set_smart, get_distances, get_squared_froebenius_norm_line_wise
from qkmeans.qk_means.kmeans import kmeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse.linalg import LinearOperator
from sklearn.svm import LinearSVC

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
    "1nn_ball_tree_accuracy",
    "nystrom_build_time",
    "nystrom_inference_time",
    "nystrom_sampled_error_reconstruction",
    "batch_assignation_mean_time",
    "nb_param_centroids",
    "nystrom_svm_accuracy",
    "nystrom_sampled_error_reconstruction_uniform",
    "nystrom_svm_time"
]

def main_kmeans(X, U_init):
    """
    Will perform the k means algorithm on X with U_init as initialization

    :param X: The input data in which to find the clusters.
    :param U_init: The initialization of the the clusters.

    :return: The final centroids, the indicator vector
    """
    start_kmeans = time.process_time()
    objective_values_k, final_centroids, indicator_vector_final = kmeans(X_data=X,
           K_nb_cluster=paraman["--nb-cluster"],
           nb_iter=paraman["--nb-iteration"],
           initialization=U_init)
    stop_kmeans = time.process_time()
    kmeans_traintime = stop_kmeans - start_kmeans

    kmeans_results = {
        "traintime": kmeans_traintime
    }

    objprinter.add("kmeans_objective", ("after t", ), objective_values_k)
    resprinter.add(kmeans_results)

    return final_centroids, indicator_vector_final


def main_qmeans(X, U_init):
    """
    Will perform the qmeans Algorithm on X with U_init as initialization.

    :param X: The input data in which to find the clusters.
    :param U_init: The initialization of the the clusters.

    :return: The final centroids as sparse factors, the indicator vector
    """
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

    start_qmeans = time.process_time()
    objective_values_q, final_centroids, indicator_vector_final = qmeans(X_data=X,
                                                 K_nb_cluster=paraman["--nb-cluster"],
                                                 nb_iter=paraman["--nb-iteration"],
                                                 nb_factors=paraman["--nb-factors"] + 1,
                                                 params_palm4msa=parameters_palm4msa,
                                                 initialization=U_init,
                                                 hierarchical_inside=paraman["--hierarchical"],
                                                 )
    stop_qmeans = time.process_time()
    qmeans_traintime = stop_qmeans - start_qmeans
    qmeans_results = {
        "traintime": qmeans_traintime
    }


    objprinter.add("qmeans_objective", ("after t", ), objective_values_q)
    resprinter.add(qmeans_results)

    return final_centroids, indicator_vector_final

def make_assignation_evaluation(X, centroids):
    nb_eval = paraman["--assignation-time"]
    if nb_eval > X.shape[0]:
        logger.warning("Batch size for assignation evaluation is bigger than data size. {} > {}. Using "
                       "data size instead.".format(nb_eval, X.shape[0]))
        nb_eval = X.shape[0]
        paraman["--assignation-time"] = nb_eval

    times = []
    precomputed_centroid_norms = get_squared_froebenius_norm_line_wise(centroids)
    for i in np.random.permutation(X.shape[0])[:nb_eval]:
        start_time = time.process_time()
        get_distances(X[i].reshape(1, -1), centroids, precomputed_centroids_norm=precomputed_centroid_norms)
        stop_time = time.process_time()
        times.append(stop_time - start_time)

    mean_time = np.mean(times)
    std_time = np.std(times)

    resprinter.add({
        "assignation_mean_time": mean_time,
        "assignation_std_time": std_time
    })


def make_batch_assignation_evaluation(X, centroids):
    """
    Assign `size_batch` random samples of `X` to some of the centroids.
    All the samples are assigned at the same time using a matrix-vector multiplication.
    Time is recorded.

    :param X: The input data from which to take the samples.
    :param centroids: The centroids to which to assign the samples (must be of same dimension than `X`)
    :param size_batch: The number of data points to assign

    :return: None
    """
    size_batch = paraman["--batch-assignation-time"]
    if size_batch > X.shape[0]:
        logger.warning("Batch size for batch assignation evaluation is bigger than data size. {} > {}. Using "
                       "data size instead.".format(size_batch, X.shape[0]))
        size_batch = X.shape[0]
        paraman["--batch-assignation-time"] = size_batch

    precomputed_centroid_norms = get_squared_froebenius_norm_line_wise(centroids)
    # precomputed_centroid_norms = None
    indexes_batch = np.random.permutation(X.shape[0])[:size_batch]
    start_time = time.process_time()
    _ = get_distances(X[indexes_batch], centroids, precomputed_centroids_norm=precomputed_centroid_norms)
    stop_time = time.process_time()

    resprinter.add({
        "batch_assignation_mean_time": (stop_time-start_time) / size_batch,
    })

def make_1nn_evaluation(x_train, y_train, x_test, y_test, U_centroids, indicator_vector):
    """
    Do the 1-nearest neighbor classification using `x_train`, `y_train` as support and `x_test`, `y_test` as
    evaluation set.

    The scikilearn classifiers (brute, kdtree and balltree) are called only in the case where it is the kmeans version
    of the program that is called (for simplicity purposes: not do it many times).

    Time is recorded.
    Classification accuracy is recorded.

    :param x_train: Train data set as ndarray.
    :param y_train: Train labels as categories in ndarray.
    :param x_test: Test data as ndarray.
    :param y_test: Test labels as categories.
    :param U_centroids: The matrix of centroids as ndarray or SparseFactor object
    :param indicator_vector: The indicator vector for this matrix of centroids and this train data.

    :return:
    """

    def scikit_evaluation(str_type):
        """
        Do the scikit learn version of nearest neighbor (used for comparison)

        :param str_type:
        :return:
        """
        clf = KNeighborsClassifier(n_neighbors=1, algorithm=str_type)
        clf.fit(x_train, y_train)
        log_memory_usage("Memory after definition of neighbors classifiers in scikit_evaluation of make_1nn_evaluation")

        start_inference_time = time.process_time()
        predictions = np.empty_like(y_test)
        for obs_idx, obs_test in enumerate(x_test):
            predictions[obs_idx] = clf.predict(obs_test.reshape(1, -1))[0]
        stop_inference_time = time.process_time()
        log_memory_usage("Memory after label assignation in scikit_evaluation of make_1nn_evaluation")

        inference_time = (stop_inference_time - start_inference_time)

        accuracy = np.sum(predictions == y_test) / y_test.shape[0]

        results_1nn = {
            "1nn_{}_inference_time".format(str_type): inference_time,
            "1nn_{}_accuracy".format(str_type): accuracy
        }
        resprinter.add(results_1nn)
        return inference_time

    def kmean_tree_evaluation():
        """
        Do the K-means partitioning version of nearest neighbor?=.

        :return:
        """
        # for each cluster, there is a sub nearest neighbor classifier for points in that cluster.
        lst_clf_by_cluster = [KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(x_train[indicator_vector == i], y_train[indicator_vector == i]) for i in range(U_centroids.shape[0])]
        log_memory_usage("Memory after definition of neighbors classifiers in kmean_tree_evaluation of make_1nn_evaluation")
        precomputed_centroid_norms = get_squared_froebenius_norm_line_wise(U_centroids)
        # precomputed_centroid_norms = None
        start_inference_time = time.process_time()
        distances = get_distances(x_test, U_centroids, precomputed_centroids_norm=precomputed_centroid_norms)
        stop_get_distances_time = time.process_time()
        get_distance_time = stop_get_distances_time - start_inference_time
        indicator_vector_test = np.argmin(distances, axis=1)
        predictions = np.empty_like(y_test)
        for obs_idx, obs_test in enumerate(x_test):
            # get the cluster to which belongs this data point and call the associated nearest neighbor classifier
            idx_cluster = indicator_vector_test[obs_idx]
            clf_cluster = lst_clf_by_cluster[idx_cluster]
            predictions[obs_idx] = clf_cluster.predict(obs_test.reshape(1, -1))[0]
        stop_inference_time = time.process_time()
        log_memory_usage("Memory after label assignation in kmean_tree_evaluation of make_1nn_evaluation")
        inference_time = (stop_inference_time - start_inference_time)

        accuracy = np.sum(predictions == y_test) / y_test.shape[0]

        results_1nn = {
            "1nn_kmean_inference_time": inference_time,
            "1nn_get_distance_time": get_distance_time / x_test.shape[0],
            "1nn_kmean_accuracy": accuracy
        }
        resprinter.add(results_1nn)
        return inference_time

    logger.info("1 nearest neighbor with k-means search")
    kmean_tree_time = kmean_tree_evaluation()
    #
    if paraman["kmeans"] and not paraman["palm"]:
        lst_knn_types = ["brute", "ball_tree", "kd_tree"]
        for knn_type in lst_knn_types:
            # the classification must not take more than 10 times the time taken for the K means 1 nn classification or
            # it will stop.
            signal.signal(signal.SIGALRM, timeout_signal_handler)
            signal.alarm(int(kmean_tree_time * 10))  # start alarm
            try:
                logger.info("1 nearest neighbor with {} search".format(knn_type))
                scikit_evaluation(knn_type)
            except TimeoutError as te:
                logger.warning("Timeout during execution of 1-nn with {} version: {}".format(knn_type, te))
            signal.alarm(0)  # stop alarm for next evaluation


def special_rbf_kernel(X, Y, gamma, norm_X, norm_Y):
    """
    Rbf kernel expressed under the form f(x)f(u)f(xy^T)

    Can handle X and Y as Sparse Factors.

    :param X: n x d matrix
    :param Y: n x d matrix
    :return:
    """
    assert len(X.shape) == len(Y.shape) == 2

    if norm_X is None:
        norm_X = get_squared_froebenius_norm_line_wise(X)
    if norm_Y is None:
        norm_Y = get_squared_froebenius_norm_line_wise(Y)

    def f(norm_mat):
        return np.exp(-gamma * norm_mat)

    def g(scal):
        return np.exp(2 * gamma * scal)

    if isinstance(X, SparseFactors) and isinstance(Y, SparseFactors):
        # xyt = SparseFactors(X.get_list_of_factors() + Y.transpose().get_list_of_factors()).compute_product(return_array=True)
        S = SparseFactors(lst_factors=X.get_list_of_factors() + Y.get_list_of_factors_H(), lst_factors_H=X.get_list_of_factors_H() + Y.get_list_of_factors())
        xyt = S.compute_product(return_array=True)
    else:
        xyt = X @ Y.transpose()

    return f(norm_X).reshape(-1, 1) * g(xyt) * f(norm_Y).reshape(1, -1)

def make_nystrom_evaluation(x_train, y_train, x_test, y_test, U_centroids):
    """
    Evaluation Nystrom construction time and approximation precision.

    The approximation is based on a subsample of size n_sample of the input data set.

    :param x_train: Input dataset as ndarray.
    :param U_centroids: The matrix of centroids as ndarray or SparseFactor object
    :param n_sample: The number of sample to take into account in the reconstruction (can't be too large)

    :return:
    """
    def prepare_nystrom(landmarks, landmarks_norm):
        basis_kernel_W = special_rbf_kernel(landmarks, landmarks, gamma, landmarks_norm, landmarks_norm)
        U, S, V = np.linalg.svd(basis_kernel_W)
        S = np.maximum(S, 1e-12)
        normalization_ = np.dot(U / np.sqrt(S), V)

        return normalization_

    def nystrom_transformation(x_input, landmarks, p_metric, landmarks_norm, x_input_norm):
        nystrom_embedding = special_rbf_kernel(landmarks, x_input, gamma, landmarks_norm, x_input_norm).T @ p_metric
        return nystrom_embedding

    n_sample = paraman["--nystrom"]
    if n_sample > x_train.shape[0]:
        logger.warning("Batch size for nystrom evaluation is bigger than data size. {} > {}. Using "
                       "data size instead.".format(n_sample, x_train.shape[0]))
        n_sample = x_train.shape[0]
        paraman["--nystrom"] = n_sample

    # Compute euristic gamma as the mean of euclidian distance between example
    gamma = compute_euristic_gamma(x_train)
    log_memory_usage("Memory after euristic gamma computation in make_nystrom_evaluation")
    # precompute the centroids norm for later use (optimization)
    centroids_norm = get_squared_froebenius_norm_line_wise(U_centroids)
    # centroids_norm = None

    indexes_samples = np.random.permutation(x_train.shape[0])[:n_sample]
    sample = x_train[indexes_samples]
    samples_norm = None
    log_memory_usage("Memory after sample selection in make_nystrom_evaluation")

    ########################
    # Nystrom on centroids #
    ########################
    logger.info("Build Nystrom on centroids")
    ## TIME: nystrom build time
    # nystrom build time is Nystrom preparation time for later use.
    ## START
    nystrom_build_start_time = time.process_time()
    metric = prepare_nystrom(U_centroids, centroids_norm)
    nystrom_build_stop_time = time.process_time()
    log_memory_usage("Memory after SVD computation in make_nystrom_evaluation")
    # STOP
    nystrom_build_time = nystrom_build_stop_time - nystrom_build_start_time

    ## TIME: nystrom inference time
    # Nystrom inference time is the time for Nystrom transformation for all the samples.
    ## START
    nystrom_inference_time_start = time.process_time()
    nystrom_embedding = nystrom_transformation(sample, U_centroids, metric, centroids_norm, samples_norm)
    nystrom_approx_kernel_value = nystrom_embedding @ nystrom_embedding.T
    nystrom_inference_time_stop = time.process_time()
    log_memory_usage("Memory after kernel matrix approximation in make_nystrom_evaluation")
    ## STOP
    nystrom_inference_time = (nystrom_inference_time_stop - nystrom_inference_time_start) / n_sample

    ################################################################

    ######################
    # Nystrom on uniform #
    ######################
    logger.info("Build Nystrom on uniform sampling")

    indexes_uniform_samples = np.random.permutation(x_train.shape[0])[:U_centroids.shape[0]]
    uniform_sample = x_train[indexes_uniform_samples]
    uniform_sample_norm = None
    log_memory_usage("Memory after uniform sample selection in make_nystrom_evaluation")

    metric_uniform = prepare_nystrom(uniform_sample, uniform_sample_norm)
    log_memory_usage("Memory after SVD computation in uniform part of make_nystrom_evaluation")

    nystrom_embedding_uniform = nystrom_transformation(sample, uniform_sample, metric_uniform, uniform_sample_norm, samples_norm)
    nystrom_approx_kernel_value_uniform = nystrom_embedding_uniform @ nystrom_embedding_uniform.T

    #################################################################

    ###############
    # Real Kernel #
    ###############
    logger.info("Compute real kernel matrix")

    real_kernel = special_rbf_kernel(sample, sample, gamma, samples_norm, samples_norm)
    real_kernel_norm = np.linalg.norm(real_kernel)
    log_memory_usage("Memory after real kernel computation in make_nystrom_evaluation")

    ################################################################

    ####################
    # Error evaluation #
    ####################

    sampled_froebenius_norm = np.linalg.norm(nystrom_approx_kernel_value - real_kernel) / real_kernel_norm
    sampled_froebenius_norm_uniform = np.linalg.norm(nystrom_approx_kernel_value_uniform - real_kernel) / real_kernel_norm

    # svm evaluation
    if x_test is not None:
        logger.info("Start classification")

        time_classification_start = time.process_time()
        x_train_nystrom_embedding = nystrom_transformation(x_train, U_centroids, metric, centroids_norm, None)
        x_test_nystrom_embedding = nystrom_transformation(x_test, U_centroids, metric, centroids_norm, None)

        linear_svc_clf = LinearSVC()
        linear_svc_clf.fit(x_train_nystrom_embedding, y_train)
        accuracy_nystrom_svm = linear_svc_clf.score(x_test_nystrom_embedding, y_test)
        time_classification_stop = time.process_time()

        delta_time_classification = time_classification_stop - time_classification_start
    else:
        accuracy_nystrom_svm = None
        delta_time_classification = None

    nystrom_results = {
        "nystrom_build_time": nystrom_build_time,
        "nystrom_inference_time": nystrom_inference_time,
        "nystrom_sampled_error_reconstruction": sampled_froebenius_norm,
        "nystrom_sampled_error_reconstruction_uniform": sampled_froebenius_norm_uniform,
        "nystrom_svm_accuracy": accuracy_nystrom_svm,
        "nystrom_svm_time": delta_time_classification
    }

    resprinter.add(nystrom_results)


def process_palm_on_top_of_kmeans(kmeans_centroids):
    lst_constraint_sets, lst_constraint_sets_desc = build_constraint_set_smart(left_dim=kmeans_centroids.shape[0],
                                                                               right_dim=kmeans_centroids.shape[1],
                                                                               nb_factors=paraman["--nb-factors"] + 1,
                                                                               sparsity_factor=paraman["--sparsity-factor"],
                                                                               residual_on_right=paraman["--residual-on-right"])

    lst_factors = init_lst_factors(*kmeans_centroids.shape, paraman["--nb-factors"] + 1)

    eye_norm = np.sqrt(kmeans_centroids.shape[0])

    if paraman["--hierarchical"]:
        _lambda_tmp, op_factors, U_centroids, nb_iter_by_factor, objective_palm = \
            hierarchical_palm4msa(
                arr_X_target=np.eye(kmeans_centroids.shape[0]) @ kmeans_centroids,
                lst_S_init=lst_factors,
                lst_dct_projection_function=lst_constraint_sets,
                f_lambda_init=1. * eye_norm,
                nb_iter=paraman["--nb-iteration-palm"],
                update_right_to_left=True,
                residual_on_right=paraman["--residual-on-right"],
                graphical_display=False)
    else:
        _lambda_tmp, op_factors, _, objective_palm, nb_iter_palm = \
            palm4msa(arr_X_target=np.eye(kmeans_centroids.shape[0]) @ kmeans_centroids,
                     lst_S_init=lst_factors,
                     nb_factors=len(lst_factors),
                     lst_projection_functions=lst_constraint_sets[-1]["finetune"],
                     f_lambda_init=1. * eye_norm,
                     nb_iter=paraman["--nb-iteration-palm"],
                     update_right_to_left=True,
                     graphical_display=False,
                     track_objective=False)

    log_memory_usage("Memory after palm on top of kmeans in process_palm_on_top_of_kmeans")

    _lambda = _lambda_tmp / eye_norm
    lst_factors_ = op_factors.get_list_of_factors()
    op_centroids = SparseFactors([lst_factors_[1] * _lambda]
                                 + lst_factors_[2:])

    return op_centroids

if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManager(arguments)
    initialized_results = dict((v, None) for v in lst_results_header)
    resprinter = ResultPrinter(output_file=paraman["--output-file_resprinter"])
    resprinter.add(initialized_results)
    resprinter.add(paraman)
    objprinter = ObjectiveFunctionPrinter(output_file=paraman["--output-file_objprinter"])
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

        dataset["x_train"] = dataset["x_train"].astype(np.float)

        sizes = {
            "size_train": dataset["x_train"].shape[0]
        }

        if "x_test" in dataset:
            dataset["x_test"] = dataset["x_test"].astype(np.float)
            dataset["y_test"] = dataset["y_test"].astype(np.float)
            dataset["y_train"] = dataset["y_train"].astype(np.float)
            sizes.update({
                "size_test": dataset["x_test"].shape[0]
            })

        resprinter.add(sizes)


        U_init = paraman.get_initialization_centroids(dataset["x_train"])

        log_memory_usage("Memory after loading dataset and initialization of centroids")

        if paraman["kmeans"]:
            U_final, indicator_vector_final = main_kmeans(dataset["x_train"], U_init)

            log_memory_usage("Memory after kmeans")

            dct_nb_param = {"nb_param_centroids": U_final.size}
            if paraman["palm"]:
                if paraman["--nb-factors"] is None:
                    paraman["--nb-factors"] = int(np.log2(min(U_init.shape)))
                paraman["--residual-on-right"] = True if U_init.shape[1] >= U_init.shape[0] else False

                U_final = process_palm_on_top_of_kmeans(U_final)
                distances = get_distances(dataset["x_train"], U_final)
                indicator_vector_final = np.argmin(distances, axis=1)
                dct_nb_param = {"nb_param_centroids": U_final.get_nb_param()}

        elif paraman["qmeans"]:
            # paraman_q = ParameterManagerQmeans(arguments)
            # paraman.update(paraman_q)
            if paraman["--nb-factors"] is None:
                paraman["--nb-factors"] = int(np.log2(min(U_init.shape)))
            paraman["--residual-on-right"] = True if U_init.shape[1] >= U_init.shape[0] else False
            U_final, indicator_vector_final = main_qmeans(dataset["x_train"], U_init)

            log_memory_usage("Memory after qmeans")

            dct_nb_param = {"nb_param_centroids": U_final.get_nb_param()}
        else:
            raise NotImplementedError("Unknown method.")
        resprinter.add(dct_nb_param)

        np.save(paraman["--output-file_centroidprinter"], U_final, allow_pickle=True)

        if paraman["--assignation-time"] is not None:
            logger.info("Start assignation time evaluation with {} samples".format(paraman["--assignation-time"]))
            make_assignation_evaluation(dataset["x_train"], U_final)

        if paraman["--batch-assignation-time"] is not None:
            logger.info("Start batch assignation time evaluation with batch size {}".format(paraman["--batch-assignation-time"]))
            make_batch_assignation_evaluation(dataset["x_train"], U_final)

        if paraman["--1-nn"] and "x_test" in dataset.keys():
            logger.info("Start 1 nearest neighbor evaluation")
            make_1nn_evaluation(x_train=dataset["x_train"],
                                y_train=dataset["y_train"],
                                x_test=dataset["x_test"],
                                y_test=dataset["y_test"],
                                U_centroids=U_final,
                                indicator_vector=indicator_vector_final)

        if paraman["--nystrom"] is not None:
            logger.info("Start Nyström reconstruction evaluation with {} samples".format(paraman["--nystrom"]))
            make_nystrom_evaluation(x_train=dataset["x_train"],
                                y_train=dataset["y_train"],
                                x_test=dataset["x_test"],
                                y_test=dataset["y_test"],
                                U_centroids=U_final)
    except Exception as e:
        has_failed = True
        failure_dict = {
            "failure": has_failed
        }

        resprinter.add(failure_dict)
        resprinter.print()
        objprinter.print()
        raise e

    failure_dict = {
        "failure": has_failed
    }

    resprinter.add(failure_dict)
    resprinter.print()
    objprinter.print()