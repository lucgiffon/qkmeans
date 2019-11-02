from qkmeans.core.utils import get_squared_froebenius_norm_line_wise
from qkmeans.kernel.kernel import prepare_nystrom, nystrom_transformation
from qkmeans.utils import kddcup99_dataset, compute_euristic_gamma
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import Nystroem
import matplotlib.pyplot as plt

if __name__ == "__main__":
    nb_clust = ["8", "16", "32", "64", "128", "256", "512"]
    nb_clust = [int(a) for a in nb_clust]
    nb_seed = 1
    sample_size = 5000
    data = kddcup99_dataset()["x_train"]

    differences_sample = np.empty((len(nb_clust), nb_seed))
    differences_total = np.empty((len(nb_clust), nb_seed))
    differences_custom_sample = np.empty((len(nb_clust), nb_seed))
    differences_custom_total = np.empty((len(nb_clust), nb_seed))
    for i in range(nb_seed):
        print("seed {}".format(i))
        np.random.seed(i)
        sample_idx = np.random.permutation(data.shape[0])[:sample_size]
        sample = data[sample_idx]
        sample_norm = None
        gamma = compute_euristic_gamma(sample)
        real_kernel_matrix = rbf_kernel(sample, gamma=gamma)

        for i_clust, msize in enumerate(nb_clust):
            print("size nys {}: {}".format(msize, i_clust))
            print("from sample")
            # nys approx sampled from sub sampled data
            sample_sample_idx = np.random.permutation(sample_size)[:msize]
            sample_sample = sample[sample_sample_idx]
            sample_sample_norm = get_squared_froebenius_norm_line_wise(sample_sample)[:, np.newaxis]

            #######################
            metric_sample = prepare_nystrom(sample_sample, sample_sample_norm, gamma=gamma)
            nystrom_embedding_sample = nystrom_transformation(sample, sample_sample, metric_sample, sample_sample_norm, sample_norm, gamma=gamma)
            K_custom_sample = nystrom_embedding_sample @ nystrom_embedding_sample.T
            dif_custom_sample = np.linalg.norm(real_kernel_matrix - K_custom_sample) / np.linalg.norm(real_kernel_matrix)
            differences_custom_sample[i_clust, i] = dif_custom_sample


            nys_sample = Nystroem(gamma=gamma, n_components=msize)
            nys_sample.fit(sample_sample)
            phi_sample = nys_sample.transform(sample)
            K_sample = phi_sample @ phi_sample.T
            dif_sample = np.linalg.norm(real_kernel_matrix - K_sample) / np.linalg.norm(real_kernel_matrix)
            differences_sample[i_clust, i] = dif_sample

            print("from total")
            # nys approx sampled from total data
            sample_total_idx = np.random.permutation(data.shape[0])[:msize]
            sample_total = data[sample_total_idx]
            sample_total_norm = get_squared_froebenius_norm_line_wise(sample_total)[:, np.newaxis]

            metric_total = prepare_nystrom(sample_total, sample_total_norm, gamma=gamma)
            nystrom_embedding_total = nystrom_transformation(sample, sample_total, metric_total, sample_total_norm, sample_norm, gamma=gamma)
            K_custom_total = nystrom_embedding_total @ nystrom_embedding_total.T
            dif_custom_total = np.linalg.norm(real_kernel_matrix - K_custom_total) / np.linalg.norm(real_kernel_matrix)
            differences_custom_total[i_clust, i] = dif_custom_total


            nys_total = Nystroem(gamma=gamma, n_components=msize)
            nys_total.fit(sample_total)
            phi_total = nys_total.transform(sample)
            K_total = phi_total @ phi_total.T

            dif_total = np.linalg.norm(real_kernel_matrix - K_total) / np.linalg.norm(real_kernel_matrix)
            differences_total[i_clust, i] = dif_total

    plt.scatter(nb_clust, np.mean(differences_sample, axis=1), label="from sample")
    plt.scatter(nb_clust, np.mean(differences_total, axis=1), label="from total")
    plt.scatter(nb_clust, np.mean(differences_custom_sample, axis=1), label="custom from sample")
    plt.scatter(nb_clust, np.mean(differences_custom_total, axis=1), label="custom from total")
    plt.legend()
    plt.show()
    # afficher ces différences pour chaque taille de clluster

    # 