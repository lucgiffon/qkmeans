from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',

        'size'   : 26}

matplotlib.rc('font', **font)

if __name__ == "__main__":
    centers = np.array([[1, 3],[3, 1],[3, 3]])
    x, y = make_blobs(centers=centers, cluster_std=0.3)
    linewidth = 2

    plt.subplot(1, 2, 1)
    plt.title("Initial observations")
    plt.scatter(x[:, 0], x[:, 1], color="grey", linewidth=linewidth)
    plt.xlabel("$\mathbf{x}_1$")
    plt.ylabel("$\mathbf{x}_2$")

    plt.subplot(1, 2, 2)
    text_offset = np.array([-0.05, -0.05])

    plt.title("Clustered observations")
    plt.scatter(x[y==0][:, 0], x[y==0][:, 1], color="r", linewidth=linewidth)
    plt.scatter(centers[0][0] + text_offset[0], centers[0][1] + text_offset[1], linewidth=linewidth, marker="x", color="r", s=100)
    plt.text(centers[0][0], centers[0][1], "$\mathbf{U}_1$")
    plt.scatter(x[y==1][:, 0], x[y==1][:, 1], color="b", linewidth=linewidth)
    plt.scatter(centers[1][0] + text_offset[0], centers[1][1] + text_offset[1], linewidth=linewidth, marker="x", color="b", s=100)
    plt.text(centers[1][0], centers[1][1], "$\mathbf{U}_2$")
    plt.scatter(x[y==2][:, 0], x[y==2][:, 1], color="g", linewidth=linewidth)
    plt.scatter(centers[2][0] + text_offset[0], centers[2][1] + text_offset[1], linewidth=linewidth, marker="x", color="g", s=100)
    plt.text(centers[2][0], centers[2][1], "$\mathbf{U}_3$")
    plt.xlabel("$\mathbf{x}_1$")
    plt.ylabel("$\mathbf{x}_2$")
    # plt.tight_layout()

    plt.show()

