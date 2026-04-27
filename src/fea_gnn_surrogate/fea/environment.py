import numpy as np

from fea_gnn_surrogate.fea import stiffness as st
from fea_gnn_surrogate.fea import member_reactions as mr
from fea_gnn_surrogate.fea import nodal_reactions as nr


class StructEnvironment:
    def __init__(self):
        self.E = 32800 * 10**6  # (N/m^2) Young's modulus
        self.current_step = 0

    def calc_A(self, d, w):
        return d * w

    def calc_I(self, d, w):
        return (w * d**3) / 12

    def analyse(self):
        members_area = self.calc_A(self.members_depth, self.members_width)
        members_moment = self.calc_I(self.members_depth, self.members_width)

        K = st.build_K(self.members, members_area, self.rotations, self.lengths, self.E, members_moment)
        U_free = nr.calc_displacement(K, self.point_loads, self.restrained_DOF)
        if U_free is None:
            return None, None, None

        UG = nr.assemble_UG(U_free, self.DOF, self.restrained_DOF)
        deflections = []
        for i in range(self.element_count):
            L = self.lengths[i]
            KG = st.calculate_Kg(0, L, self.E, members_area[i], members_moment[i])
            KG = st.calc_KG_to_mat(KG)
            U_local = mr.transform_U_to_local(UG, self.members[i], self.rotations[i]).ravel()
            F_local = (KG @ U_local).ravel()
            deflection_x = mr.calc_deflection_x_diagram_numerical(L, self.step_sizes[i], U_local)
            deflection_y = mr.calc_deflection_diagram_analytical(
                F_local, self.E * members_moment[i], L, self.step_sizes[i], U_local
            )
            deflections.append(np.vstack((deflection_x, deflection_y)).T)
        return UG, deflections, K

    def set_attributes(
        self,
        node_positions,
        node_restrained_dof,
        node_loads,
        edges,
        edge_rotations,
        edge_lenghts,
        edge_depths,
        edge_widths,
        step_sizes,
    ):
        self.nodes = node_positions
        self.members = edges
        self.restrained_DOF = node_restrained_dof
        self.DOF = st.calc_DOF(edges)
        self.element_count = len(edges)
        self.point_loads = node_loads
        self.rotations = edge_rotations
        self.lengths = edge_lenghts
        self.members_depth = edge_depths
        self.members_width = edge_widths
        self.step_sizes = step_sizes

    def calc_drift(self, U, num_cols, num_rows):
        U = U.reshape(num_cols, num_rows, 3)
        drift = np.abs(np.diff(U[:, :, 0])).max(axis=0)
        return drift
