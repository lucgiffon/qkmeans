import numpy as np
from pyqalm.data_structures import SparseFactors
from pyqalm.utils import constant_proj, get_lambda_proxsplincol


def get_distances(X_data, centroids):
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
    if isinstance(centroids, SparseFactors):
        mat_centroids = centroids.compute_product(return_array=False)
        centroid_norms = np.linalg.norm(mat_centroids.toarray(), axis=1) ** 2
        # centroid_norms = np.sqrt(centroids.power(2).sum(axis=1))
    else:
        centroid_norms = np.linalg.norm(centroids, axis=1) ** 2

    centroid_distances = centroid_norms[:, None] - 2 * centroids @ X_data.T

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


def assign_points_to_clusters(X, centroids):
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
    distances = get_distances(X, centroids)
    # then, Determine class membership of each point
    # by picking the closest centroid
    indicator_vector = np.argmin(distances, axis=1)
    return indicator_vector


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
                local_lst_constraints.append(lambda mat: mat)
            else:
                local_lst_constraints.append(get_lambda_proxsplincol(val))

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