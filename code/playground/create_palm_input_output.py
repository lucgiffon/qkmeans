import numpy as np

from qkmeans.core.qmeans_fast import init_lst_factors
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.palm.palm_fast import palm4msa

from pprint import pprint
from pathlib import Path


d = 16
n_facs = [2, 4]
sparsity_fac = 2
nb_iter_palm= 300
delta_objective = 1e-6
update_right_to_left = True
dims = [(d, d), (d, d//2), (d//2, d)]

results = {}
for n_fac in n_facs:
    for dim in dims:
        pair = dict()
        lst_factors = init_lst_factors(dim[0], dim[1], n_fac, first_square=False)

        # construit les contraintes de projection dans une liste
        lst_constraints, lst_constraints_vals = build_constraint_set_smart(
            dim[0], dim[1], n_fac,
            sparsity_factor=sparsity_fac, residual_on_right=True,
            fast_unstable_proj=False, constant_first=False)
        lst_constraints_palm = lst_constraints[-1]["finetune"]

        # construit la matrice cible
        X_target = np.random.rand(dim[0],  dim[1])

        # op_factor est en quelque sortes la liste des facteurs sparses
        _lambda, op_factors, _, _, _ = \
            palm4msa(
                arr_X_target=X_target,
                lst_S_init=lst_factors,
                nb_factors=len(lst_factors),
                lst_projection_functions=lst_constraints_palm,
                f_lambda_init=1.,
                nb_iter=nb_iter_palm,
                update_right_to_left=update_right_to_left,
                track_objective=True,
                delta_objective_error_threshold=delta_objective)

        pair["input"] = X_target
        pair["output"] = _lambda * op_factors.compute_product(return_array=True)

        results["nfac_{}_in_{}_out_{}".format(n_fac, dim[0], dim[1])] = pair

path_dir = Path(__file__.split(".")[0]) / "examples_jovial"
path_dir.mkdir(parents=True, exist_ok=True)

for name_xp, dct_xp in results.items():
    path_file = path_dir / (name_xp + ".npz")
    np.savez(str(path_file), **dct_xp)