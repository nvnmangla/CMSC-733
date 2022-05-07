import numpy as np



def get_essential_matrix(F_matrix, K_matrix):
    E_matrix = np.dot(K_matrix.T, np.dot(F_matrix, K_matrix))
    U, S, V_T = np.linalg.svd(E_matrix)
    S = np.diag(S)
    S[2, 2] = 0
    E_matrix = np.dot(U, np.dot(S, V_T))
    return E_matrix