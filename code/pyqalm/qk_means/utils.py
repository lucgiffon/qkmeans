import numpy as np
from pyqalm.data_structures import SparseFactors
from pyqalm.utils import constant_proj, get_lambda_proxsplincol, logger, DataGenerator


def get_squared_froebenius_norm_line_wise(data_arr):
    if isinstance(data_arr, SparseFactors):
        mat_centroids = data_arr.compute_product(return_array=True)
        centroid_norms = np.linalg.norm(mat_centroids, axis=1) ** 2
        # centroid_norms = np.sqrt(centroids.power(2).sum(axis=1))
    else:
        centroid_norms = np.linalg.norm(data_arr, axis=1) ** 2

    return centroid_norms

def get_squared_froebenius_norm_line_wise_batch_by_batch(data_arr_memmap, batch_size):
    data_norms = np.zeros(data_arr_memmap.shape[0])
    logger.debug("Start computing norm of datat array of shape {}, batch by batch".format(data_arr_memmap.shape))
    for i_batch, batch in enumerate(DataGenerator(data_arr_memmap, batch_size=batch_size, return_indexes=False)):
        logger.debug("Compute norm of batch {}/{}".format(i_batch, data_arr_memmap.shape[0]//batch_size))
        data_norms[i_batch*batch_size:(i_batch+1)*batch_size] = np.linalg.norm(batch, axis=1) ** 2
    return data_norms

def get_distances(X_data, centroids, precomputed_centroids_norm=None, precomputed_data_points_norm=None):
    """
    Return the matrix of distance between each data point and each centroid.

    Parameters
    ----------
    X_data : np.ndarray [n, d]
    centroids : np.ndarray or SparseFactors [k, d]

    Returns
    -------
    np.ndarray [k, n]
    """
    if precomputed_centroids_norm is not None:
        centroid_norms = precomputed_centroids_norm
    else:
        centroid_norms = get_squared_froebenius_norm_line_wise(centroids)

    if precomputed_data_points_norm is not None:
        data_point_norms = precomputed_data_points_norm
    else:
        data_point_norms = get_squared_froebenius_norm_line_wise(X_data)

    centroid_distances = centroid_norms[:, None] - 2 * centroids @ X_data.T + data_point_norms[None, :]

    return centroid_distances.T


def compute_objective(X_data, centroids, indicator_vector):
    """
    Compute K-means objective function

    Parameters
    ----------
    X_data : np.ndarray [n, d]
    centroids : np.ndarray or SparseFactors [k, d]
    indicator_vector : np.ndarray [n]

    Returns
    -------
    float
    """
    if isinstance(centroids, SparseFactors):
        centroids = centroids.compute_product()
    return np.linalg.norm(X_data - centroids[indicator_vector, :]) ** 2


def assign_points_to_clusters(X, centroids, X_norms=None):
    """

    Parameters
    ----------
    X : np.ndarray [n, d]
    centroids : np.ndarray or SparseFactors [k, d]

    Returns
    -------
    np.ndarray [n]
        indicator_vector
    """

    # Assign all points to the nearest centroid
    # first get distance from all points to all centroids
    distances = get_distances(X, centroids, precomputed_data_points_norm=X_norms)
    # then, Determine class membership of each point
    # by picking the closest centroid
    indicator_vector = np.argmin(distances, axis=1)
    return indicator_vector, distances


def build_constraint_set_smart(left_dim, right_dim, nb_factors, sparsity_factor, residual_on_right):
    """

    :param left_dim:
    :param right_dim:
    :param nb_factors: the number of total factors, including the extra diagonal of sqrt(count).
    :param sparsity_factor:
    :param residual_on_right:
    :return:
    """
    def build_lst_constraint_from_values(lst_values):
        local_lst_constraints = []
        for val in lst_values:
            if val == "constant_proj":
                local_lst_constraints.append(constant_proj)
            elif val == "ident":
                lambda_func = lambda mat: mat
                lambda_func.__name__ = "ident"
                local_lst_constraints.append(lambda_func)
            else:
                lambda_func = get_lambda_proxsplincol(val)
                lambda_func.__name__ = "proxsplincol_{}".format(val)
                local_lst_constraints.append(lambda_func)

        return  local_lst_constraints

    def build_constraint_split(p_nb_keep_values, p_nb_keep_values_residual, p_index, p_nb_keep_values_left_most, p_nb_keep_values_right_most):
        if residual_on_right:
            if p_index == 0:
                lst_values = ["constant_proj", "ident"]
            elif p_index == 1:
                lst_values = [p_nb_keep_values_left_most, p_nb_keep_values_residual]
            elif p_index == nb_factors-2:
                lst_values = [p_nb_keep_values_left_most, max(p_nb_keep_values_right_most, p_nb_keep_values_residual)]
            else:
                lst_values = [p_nb_keep_values, p_nb_keep_values_residual]
        else:
            if p_index == nb_factors-2:
                lst_values = ["constant_proj"] + [p_nb_keep_values_left_most]
            elif p_index == 0:
                lst_values = [p_nb_keep_values_residual, p_nb_keep_values_right_most]
            else:
                lst_values = [p_nb_keep_values_residual, p_nb_keep_values]

        return build_lst_constraint_from_values(lst_values), lst_values

    def build_constraint_finetune(p_nb_keep_values, p_nb_keep_values_residual, p_index, p_nb_keep_values_left_most, p_nb_keep_values_right_most):
        if residual_on_right:
            if p_index == 0:
                lst_values = ["constant_proj", "ident"]
            elif p_index == 1:
                lst_values = ["constant_proj"] + [p_nb_keep_values_left_most, p_nb_keep_values_residual]
            elif p_index == nb_factors-2:
                lst_values = ["constant_proj"] + [p_nb_keep_values_left_most] + [p_nb_keep_values] * (p_index-1) + [max(p_nb_keep_values_right_most, p_nb_keep_values_residual)]
            else:
                lst_values = ["constant_proj"] + [p_nb_keep_values_left_most] + [p_nb_keep_values] * (p_index-1) + [p_nb_keep_values_residual]
        else:
            if p_index == nb_factors - 2:
                lst_values = ["constant_proj"] + [p_nb_keep_values_left_most] + [p_nb_keep_values] * (p_index-1) + [p_nb_keep_values_right_most]
            elif p_index == 0:
                lst_values = [p_nb_keep_values_residual, p_nb_keep_values_right_most]
            else:
                lst_values = [p_nb_keep_values_residual] + [p_nb_keep_values] * (p_index) + [p_nb_keep_values_right_most]

        return build_lst_constraint_from_values(lst_values), lst_values

    inner_factor_dim = min(left_dim, right_dim)

    lst_proj_op_by_fac_step = []
    lst_proj_op_desc_by_fac_step = []

    nb_keep_values = sparsity_factor * inner_factor_dim
    nb_keep_values_left_most = int(nb_keep_values * left_dim / inner_factor_dim)
    nb_keep_values_right_most = int(nb_keep_values * right_dim / inner_factor_dim)


    for k in range(nb_factors - 1):

        if residual_on_right:
            # power k instead of (k+1) for the first, constant matrix
            nb_values_residual = max(nb_keep_values, int(right_dim * inner_factor_dim / 2 ** (k)))
        else:
            nb_values_residual = max(nb_keep_values, int(left_dim * inner_factor_dim / 2 ** (k+1)))

        constraints_split, constraints_split_desc = build_constraint_split(nb_keep_values, nb_values_residual, k, nb_keep_values_left_most, nb_keep_values_right_most)
        constraints_finetune, constraints_finetune_desc = build_constraint_finetune(nb_keep_values, nb_values_residual, k, nb_keep_values_left_most, nb_keep_values_right_most)

        dct_step_lst_proj_op = {
            "split": constraints_split,
            "finetune": constraints_finetune
        }

        dct_step_lst_nb_keep_values = {
            "split": constraints_split_desc,
            "finetune": constraints_finetune_desc
        }

        lst_proj_op_by_fac_step.append(dct_step_lst_proj_op)
        lst_proj_op_desc_by_fac_step.append(dct_step_lst_nb_keep_values)

    return lst_proj_op_by_fac_step, lst_proj_op_desc_by_fac_step


def build_constraint_sets(left_dim, right_dim, nb_factors, sparsity_factor):
    """
    Build constraint set for factors with first factor constant.

    :param left_dim:
    :param right_dim:
    :param nb_factors:
    :param sparsity_factor:
    :return:
    """
    raise DeprecationWarning("should use build constraint set smart instead")

    inner_factor_dim = min(left_dim, right_dim)

    lst_proj_op_by_fac_step = []
    lst_proj_op_desc_by_fac_step = []

    nb_keep_values = sparsity_factor * inner_factor_dim  # sparsity factor = 5
    for k in range(nb_factors - 1):
        nb_values_residual = max(nb_keep_values, int(inner_factor_dim / 2 ** (k)) * inner_factor_dim)  # k instead of (k+1) for the first, constant matrix
        if k == 0:
            dct_step_lst_proj_op = {
                "split": [constant_proj, lambda mat: mat],
                "finetune": [constant_proj, lambda mat: mat]
            }
            dct_step_lst_nb_keep_values = {
                "split": ["constant_proj", "ident"],
                "finetune": ["constant_proj", "ident"]
            }
        else:
            dct_step_lst_proj_op = {
                "split": [get_lambda_proxsplincol(nb_keep_values), get_lambda_proxsplincol(nb_values_residual)],
                "finetune": [constant_proj] + [get_lambda_proxsplincol(nb_keep_values)] * (k) + [get_lambda_proxsplincol(nb_values_residual)]
            }

            dct_step_lst_nb_keep_values = {
                "split": [nb_keep_values, nb_values_residual],
                "finetune": ["constant_proj"] + [nb_keep_values] * (k) + [nb_values_residual]
            }

        lst_proj_op_by_fac_step.append(dct_step_lst_proj_op)
        lst_proj_op_desc_by_fac_step.append(dct_step_lst_nb_keep_values)

    return lst_proj_op_by_fac_step, lst_proj_op_desc_by_fac_step


def update_clusters_with_integrity_check(X_data, X_data_norms, X_centroids_hat, K_nb_cluster, counts, indicator_vector, distances, cluster_names, cluster_names_sorted):
    """
    Checki if no cluster has lost point and if yes, create a new cluster with the farthest point away in the cluster with the biggest population.

    All changes are made in place but for counts and cluster_names_sorted which are returned.

    :param X_data:
    :param X_data_norms:
    :param X_centroids_hat:
    :param K_nb_cluster:
    :param counts:
    :param indicator_vector:
    :param distances:
    :param cluster_names:
    :param cluster_names_sorted:
    :return:
    """

    for c in range(K_nb_cluster):
        biggest_cluster_index = np.argmax(counts)  # type: int
        biggest_cluster = cluster_names[biggest_cluster_index]
        biggest_cluster_data_indexes = indicator_vector == biggest_cluster
        index_of_farthest_point_in_biggest_cluster = np.argmax(distances[:, c][biggest_cluster_data_indexes])
        farthest_point_in_biggest_cluster = X_data[biggest_cluster_data_indexes][index_of_farthest_point_in_biggest_cluster]
        absolute_index_of_farthest_point_in_biggest_cluster = np.where(biggest_cluster_data_indexes)[0][index_of_farthest_point_in_biggest_cluster]

        cluster_data = X_data[indicator_vector == c]
        if len(cluster_data) == 0:
            logger.warning("cluster has lost data, add new cluster. cluster idx: {}".format(c))
            X_centroids_hat[c] = farthest_point_in_biggest_cluster.reshape(1, -1)
            counts = list(counts)
            counts[biggest_cluster_index] -= 1
            counts.append(1)
            counts = np.array(counts)
            cluster_names_sorted = list(cluster_names_sorted)
            cluster_names_sorted.append(c)
            cluster_names_sorted = np.array(cluster_names_sorted)

            indicator_vector[absolute_index_of_farthest_point_in_biggest_cluster] = c
            distances_to_new_cluster = get_distances(X_data, X_centroids_hat[c].reshape(1, -1), precomputed_data_points_norm=X_data_norms)
            distances[:, c] = distances_to_new_cluster.flatten()
        else:
            X_centroids_hat[c] = np.mean(X_data[indicator_vector == c, :], 0)

    return counts, cluster_names_sorted

def update_clusters(X_data, X_centroids_hat, K_nb_cluster, counts_before, new_counts, indicator_vector):
    """
    Update centroids and return new counts of each centroid.
    All changes are made in place.

    :param X_data:
    :param X_data_norms:
    :param X_centroids_hat:
    :param K_nb_cluster:
    :param new_counts:
    :param indicator_vector:
    :param distances:
    :param cluster_names:
    :param cluster_names_sorted:
    :return:
    """
    total_count_vector = counts_before + new_counts
    for c in range(K_nb_cluster):
        if total_count_vector[c] != 0:
            X_centroids_hat[c] = ((counts_before[c] / total_count_vector[c]) * X_centroids_hat[c]) +  ((1. / total_count_vector[c]) * np.sum(X_data[indicator_vector == c, :], 0))
        else:
            logger.warning("Cluster {} has zero point, continue".format(c))

    return total_count_vector

def check_cluster_integrity(X_data, X_centroids_hat, K_nb_cluster, counts, indicator_vector):
    """
    Check for each cluster if it has data points in it. If not, create a new cluster from the data points of the most populated cluster so far.

    :param X_data:
    :param X_centroids_hat:
    :param K_nb_cluster:
    :param counts:
    :param indicator_vector:
    :return:
    """
    for c in range(K_nb_cluster):

        cluster_data = X_data[indicator_vector == c]
        if len(cluster_data) == 0:
            biggest_cluster_index = np.argmax(counts)  # type: int
            biggest_cluster_data_indexes_bool = indicator_vector == biggest_cluster_index
            biggest_cluster_actual_data_indexes = np.where(biggest_cluster_data_indexes_bool)[0]

            random_index_in_biggest_cluster = np.random.choice(biggest_cluster_actual_data_indexes, size=1)[0]
            random_point_in_biggest_cluster = X_data[random_index_in_biggest_cluster]

            logger.warning("cluster has lost data, add new cluster. cluster idx: {}".format(c))
            X_centroids_hat[c] = random_point_in_biggest_cluster.reshape(1, -1)
            counts[biggest_cluster_index] -= 1
            counts[c] = 1

            indicator_vector[random_index_in_biggest_cluster] = c

