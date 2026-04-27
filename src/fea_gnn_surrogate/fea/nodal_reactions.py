import numpy as np
import copy


def build_matrix_reduced(matrix, restrained_DOF):
    matrix = np.delete(matrix, restrained_DOF, 0)
    if len(matrix.shape) > 1 and matrix.shape[1] > 1:
        matrix = np.delete(matrix, restrained_DOF, 1)
    return matrix


def calc_displacement(K, force_vector, restrained_DOF):
    Kr = build_matrix_reduced(K, restrained_DOF)
    force_vector_restrained = copy.copy(force_vector)
    force_vector_restrained = np.delete(force_vector_restrained, restrained_DOF, 0)

    try:
        U = np.matmul(np.linalg.inv(Kr), force_vector_restrained).ravel()
    except np.linalg.LinAlgError:
        print("singular matrix")
        return None
    return U


def assemble_UG(U, n_DOF, restrained_DOF):
    UG = np.zeros(n_DOF)
    c = 0
    for i in np.arange(n_DOF):
        if i in restrained_DOF:
            UG[i] = 0
        else:
            UG[i] = U[c]
            c = c + 1
    return UG


def calc_reaction(UG, K):
    FG = np.matmul(K, UG)
    return FG
