import math
import numpy as np


def calculate_Kg(theta, L, E, A, I):
    """Calculate the global stiffness matrix for a frame element."""
    c = math.cos(theta)
    s = math.sin(theta)
    c2 = c**2
    s2 = s**2
    L2 = L**2

    K11 = (E / L) * np.array(
        [
            [A * c2 + (12 * I * s2) / L2, s * c * (A - 12 * I / L2), -6 * I * s / L],
            [s * c * (A - 12 * I / L2), A * s2 + (12 * I * c2) / L2, 6 * I * c / L],
            [-6 * I * s / L, 6 * I * c / L, 4 * I],
        ]
    )

    K12 = (E / L) * np.array(
        [
            [-(A * c2 + (12 * I * s2) / L2), -s * c * (A - 12 * I / L2), -6 * I * s / L],
            [-s * c * (A - 12 * I / L2), -(A * s2 + (12 * I * c2) / L2), 6 * I * c / L],
            [6 * I * s / L, -6 * I * c / L, 2 * I],
        ]
    )

    K21 = K12.T

    K22 = (E / L) * np.array(
        [
            [A * c2 + (12 * I * s2) / L2, s * c * (A - 12 * I / L2), 6 * I * s / L],
            [s * c * (A - 12 * I / L2), A * s2 + (12 * I * c2) / L2, -6 * I * c / L],
            [6 * I * s / L, -6 * I * c / L, 4 * I],
        ]
    )

    return [K11, K12, K21, K22]


def calc_DOF(members):
    DOF = len(set(members.flatten())) * 3
    return DOF


def build_K(members, members_area, orientations, lengths, E, members_moment):
    n_DOF = calc_DOF(members)
    Kp = np.zeros([n_DOF, n_DOF])
    for n, mbr in enumerate(members):
        theta = orientations[n]
        L = lengths[n]
        A = members_area[n]
        I = members_moment[n]
        [K11, K12, K21, K22] = calculate_Kg(theta, L, E, A, I)

        node_i = mbr[0]
        node_j = mbr[1]

        i = 3 * node_i
        j = 3 * node_j

        Kp[i : i + 3, i : i + 3] = Kp[i : i + 3, i : i + 3] + K11
        Kp[j : j + 3, j : j + 3] = Kp[j : j + 3, j : j + 3] + K22
        Kp[i : i + 3, j : j + 3] = Kp[i : i + 3, j : j + 3] + K12
        Kp[j : j + 3, i : i + 3] = Kp[j : j + 3, i : i + 3] + K21

    return Kp


def extract_member_K(member, K):
    node_i = member[0]
    node_j = member[1]

    i = 3 * node_i
    j = 3 * node_j

    k_local = np.zeros((6, 6))

    k_local[0:3, 0:3] = K[i : i + 3, i : i + 3]
    k_local[3:6, 3:6] = K[j : j + 3, j : j + 3]
    k_local[0:3, 3:6] = K[i : i + 3, j : j + 3]
    k_local[3:6, 0:3] = K[j : j + 3, i : i + 3]

    return k_local


def calc_KG_to_mat(KG):
    K = np.zeros((6, 6))
    K[:3, :3] = KG[0]
    K[3:, 3:] = KG[3]
    K[:3, 3:] = KG[1]
    K[3:, :3] = KG[2]
    return K
