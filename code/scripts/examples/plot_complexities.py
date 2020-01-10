import numpy as np
import matplotlib.pyplot as plt

def kmeans_complexity(N, K, D):
    return N * K * D +  N * D

def qkmeans_complexity(N, K, D):
    A = min(K, D)
    B = max(K, D)
    return N * (A * np.log2(A) + B) + np.log2(B) * ((A * np.log2(B) + B + A*B*np.log2(B) + A**2 * np.log(A) + A**2 * np.log2(B)) + A * B + A *B) * 300 + N * D + K * D


def main():
    Ns = np.linspace(0, 1e6, num=int(1e2))
    K = 32
    D = 32
    kmeans = np.array(list(map(lambda x: kmeans_complexity(x, K, D), Ns)))
    qkmeans = np.array(list(map(lambda x: qkmeans_complexity(x, K, D), Ns)))
    plt.plot(Ns, kmeans, label="kmeans")
    plt.plot(Ns, qkmeans, label="qkmeans")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()