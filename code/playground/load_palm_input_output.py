import numpy as np


path_to_file = "/home/luc/PycharmProjects/qalm_qmeans/code/playground/create_palm_input_output/examples_jovial/nfac_2_in_8_out_16.npz"

dict_matrices = np.load(str(path_to_file))
matrice_target = dict_matrices["input"]
print(matrice_target.shape)
matrice_obtenue = dict_matrices["output"]
print(matrice_obtenue.shape)