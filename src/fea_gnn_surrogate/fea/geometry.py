import numpy as np
import math


def member_rotation(member, nodes):
    node_i = member[0]
    node_j = member[1]

    xi = nodes[node_i][0]
    yi = nodes[node_i][1]
    xj = nodes[node_j][0]
    yj = nodes[node_j][1]

    dx = xj - xi
    dy = yj - yi
    L = math.sqrt(dx**2 + dy**2)
    member_vector = np.array([dx, dy])

    if dx > 0 and dy == 0:
        theta = 0
    elif dx == 0 and dy > 0:
        theta = math.pi / 2
    elif dx < 0 and dy == 0:
        theta = math.pi
    elif dx == 0 and dy < 0:
        theta = 3 * math.pi / 2
    elif dx > 0 and dy > 0:
        ref_vector = np.array([1, 0])
        theta = math.acos(ref_vector.dot(member_vector) / (L))
    elif dx < 0 and dy > 0:
        ref_vector = np.array([0, 1])
        theta = (math.pi / 2) + math.acos(
            ref_vector.dot(member_vector) / (L)
        )
    elif dx < 0 and dy < 0:
        ref_vector = np.array([-1, 0])
        theta = math.pi + math.acos(
            ref_vector.dot(member_vector) / (L)
        )
    else:
        ref_vector = np.array([0, -1])
        theta = (3 * math.pi / 2) + math.acos(
            ref_vector.dot(member_vector) / (L)
        )

    return [theta, L]


def calc_rotation_length(members, nodes):
    rotations = np.array([])
    lengths = np.array([])
    for member in members:
        [angle, length] = member_rotation(member, nodes)
        rotations = np.append(rotations, angle)
        lengths = np.append(lengths, length)
    return rotations, lengths
