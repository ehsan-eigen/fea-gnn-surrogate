import numpy as np
import math


def transform_F_to_local(member, theta, K_member, U, PL):
    node_i = member[0]
    node_j = member[1]

    i = 3 * node_i
    j = 3 * node_j

    c = math.cos(theta)
    s = math.sin(theta)

    T = np.array(
        [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    U_member = np.array([[U[i], U[i + 1], U[i + 2], U[j], U[j + 1], U[j + 2]]]).ravel()
    PL_member = np.array([[PL[i], PL[i + 1], PL[i + 2], PL[j], PL[j + 1], PL[j + 2]]]).ravel()

    a = K_member @ U_member
    b = -(a - PL_member)
    c = T @ b
    return c


def transform_U_to_local(vector, member, theta):
    node_i = member[0]
    node_j = member[1]

    i = 3 * node_i
    j = 3 * node_j

    c = math.cos(theta)
    s = math.sin(theta)

    T = np.array(
        [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    vector = np.array(
        [[vector[i], vector[i + 1], vector[i + 2], vector[j], vector[j + 1], vector[j + 2]]]
    ).T

    local_vector = np.matmul(T, vector)
    return local_vector


def transform_V_to_local(v, theta):
    c = math.cos(theta)
    s = math.sin(theta)

    T = np.array(
        [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    local_vector = np.matmul(T, v)
    return local_vector


def inverse_rotate_vector(vector, theta):
    c = math.cos(theta)
    s = math.sin(theta)

    T = np.array(
        [
            [c, -s],
            [s, c],
        ]
    ).T

    rotated_vector = np.matmul(vector, T)
    return rotated_vector


def calc_shear_diagram(F, L, step_size):
    bin_count = int(L / step_size)
    SD = np.ones(bin_count) * F[1]
    return SD


def calc_moment_diagram(F, L, step_size):
    bin_count = int(L / step_size)
    X = [k * step_size for k in range(bin_count)]
    a = (F[5] - F[2]) / (L - step_size)
    MD = np.array([F[2] + X[i] * a for i in range(len(X))]).flatten()
    return MD


def calc_deflection_y_diagram_numerical(MD, EI, L, step_size, U):
    bin_count = int(L / step_size)

    rotation = np.zeros(bin_count)
    rotation[0] = U[2]
    deflection = np.zeros(bin_count)
    deflection[0] = U[1]

    M_im1 = MD[0]
    rotation_im1 = rotation[0]
    V_im1 = deflection[0]

    for i in range(1, len(MD)):
        rotation[i] = np.cumsum(MD[: i + 1])[-1] / EI * step_size + rotation[0]
        deflection[i] = (
            V_im1 + 0.5 * (rotation[i] + rotation_im1) * step_size
        )

        rotation_im1 = rotation[i]
        V_im1 = deflection[i]

    return deflection


def calc_deflection_x_diagram_numerical(L, step_size, U):
    bin_count = int(L / step_size)
    deflection = np.zeros(bin_count)
    deflection[0] = U[0]
    slope = (U[3] - U[0]) / L
    for i in range(bin_count):
        deflection[i] = slope * (i * step_size) + U[0]
    return deflection


def calc_deflection_diagram_analytical(F, EI, L, step_size, U):
    bin_count = int(L / step_size)
    deflection = [
        U[1]
        + U[2] * ((i * step_size))
        + 1 / EI * (-1 / 2 * F[2] * (i * step_size) ** 2 + 1 / 6 * F[1] * (i * step_size) ** 3)
        for i in range(bin_count)
    ]
    return np.array(deflection).ravel()


def calc_axial_force_from_displacement(L, U, E, member_area, node_i, node_j):
    L_new = np.sqrt((node_j[0] + U[3] - node_i[0] - U[0]) ** 2 + (node_j[1] + U[4] - node_i[1] - U[1]) ** 2)
    axial_disp = L - L_new
    F_axial = (member_area * E / L) * axial_disp
    return F_axial


def extract_member_vector(member, v):
    node_i = member[0]
    node_j = member[1]

    i = 3 * node_i
    j = 3 * node_j

    vector_member = np.array([[v[i], v[i + 1], v[i + 2], v[j], v[j + 1], v[j + 2]]]).ravel()
    return vector_member
