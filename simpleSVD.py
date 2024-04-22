import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(matrix):
    plt.imshow(matrix, cmap='binary', interpolation='nearest')
    plt.show()

# Example matrix
matrix = np.array([[1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
 [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
 [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
 [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]])

for k in [1, 2, 3, 4]:
    # Perform SVD
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)


    U_k = U[:, :k]
    s_k = np.diag(s[:k])
    Vt_k = Vt[:k, :]

    # Reconstruct the compressed matrix
    compressed_matrix = np.dot(U_k, np.dot(s_k, Vt_k))

    print("Compressed matrix with k = {k}:")
    plot_matrix(compressed_matrix)