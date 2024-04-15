#refrence from https://www.accel.ai/anthology/2022/8/17/svd-algorithm-tutorial-in-python
import numpy as np
from scipy.linalg import svd

#V tansposed matrix
def calcVT(M):
    # Compute VT matrix
    newM = np.dot(M.T, M)
    eigenvalues, eigenvectors = np.linalg.eig(newM)
    idx = np.argsort(eigenvalues)[::-1]
    V = eigenvectors[:, idx].T

    return V

def calcU(M):
    # Compute U matrix
    newM = np.dot(M, M.T)
    eigenvalues, eigenvectors = np.linalg.eig(newM)
    ncols = np.argsort(eigenvalues)[::-1] 
    U = eigenvectors[:, ncols]

    return U

#Function that calculates Eigenvalues corresponding to the Sigma Matrix
def calcD(M):
    if np.size(np.dot(M, M.T)) > np.size(np.dot(M.T, M)):
        newM = np.dot(M.T, M)
    else:
        newM = np.dot(M, M.T)

    eigenvalues = np.linalg.eig(newM)
    eigenvalues = np.sqrt(eigenvalues[0]) 
    return np.sort(eigenvalues)[::-1]



A = np.array([[3,4,3],[1,1,-5]])
VT = calcVT(A)
U = calcU(A)
S = calcD(A)

print (VT, "\n")
print(U,"\n")
print(S)