import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import math


def _laplacian_pe(edge_index, num_nodes, k=8):
    """Compute Laplacian Positional Encoding for each node.

    Returns an [N, k] tensor where each node gets k spectral coordinates.
    """
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    A = np.zeros((num_nodes, num_nodes))
    A[row, col] = 1.0
    A = np.maximum(A, A.T)  # symmetrize
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L = np.eye(num_nodes) - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    _, vecs = np.linalg.eigh(L)
    pe = vecs[:, 1 : k + 1]  # skip trivial eigenvector 0
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return torch.tensor(pe, dtype=torch.float)


class GraphHandler:
    def __init__(self, conf):
        self.num_cols = conf["num_cols"]
        self.num_rows = conf["num_rows"]
        self.transfer_row = conf["transfer_row"]
        self.horizontal_scale = conf["horizontal_scale"]
        self.vertical_scale = conf["vertical_scale"]
        self.possible_columns_up = conf.get("possible_columns_up")
        self.possible_columns_down = conf.get("possible_columns_down")
        self.distance_lower_bound = conf.get("distance_lower_bound", 4)
        self.distance_upper_bound = conf.get("distance_upper_bound", 10)

    def generate_graph(self, mode="train"):
        G = nx.grid_2d_graph(self.num_cols, self.num_rows)
        self.set_coo(G)
        self.set_pos(G)
        sample_columns_up = self.sample_cols(self.possible_columns_up)
        sample_columns_down = self.sample_cols(self.possible_columns_down)

        self.remove_cols(G, sample_columns_up, sample_columns_down)
        self.remove_foundation_beams(G)
        self.remove_disconnected_nodes(G)
        G = GraphHandler.node_tuple_2_index(G)
        self.set_edge_size(G)
        self.set_DOF(G)
        self.set_foundation_flag(G)
        self.set_loads(G)
        self.set_rotation(G)
        self.set_column_flag(G)
        self.set_distance(G)
        self.set_cant_flag(G)
        self.set_step_size(G)
        return G

    def simplify_graph(self, G, use_hop_edges=False):
        Gs = self.unify_edges(G.copy())
        if use_hop_edges:
            self.add_hop_edges(Gs)
        else:
            self.set_real_flag(Gs)
        self.set_rotation(Gs)
        self.set_column_flag(Gs)
        self.set_distance(Gs)
        self.set_edge_size(Gs)
        self.set_foundation_flag(Gs)
        self.set_transfer_flag(Gs)
        self.set_cant_flag(Gs)
        self.set_step_size(Gs)
        self.set_level(Gs)
        self.set_column_l2r_ratio(Gs)
        return Gs

    def set_pos(self, G):
        for node in G.nodes():
            (x, y) = node
            G.nodes[node]["pos"] = [x * self.horizontal_scale, y * self.vertical_scale]

    def set_edge_size(self, G):
        for edge in G.edges():
            (x1, y1) = G.nodes[edge[0]]["coo"]
            (x2, y2) = G.nodes[edge[1]]["coo"]
            if y1 == y2 and y1 == self.transfer_row:
                G[edge[0]][edge[1]]["D"] = 0.8
                G[edge[0]][edge[1]]["W"] = 0.4
            elif y1 == y2 and y1 > self.transfer_row:
                G[edge[0]][edge[1]]["D"] = 0.3
                G[edge[0]][edge[1]]["W"] = 0.4
            elif y1 == y2 and y1 < self.transfer_row:
                G[edge[0]][edge[1]]["D"] = 0.3
                G[edge[0]][edge[1]]["W"] = 0.4
            elif x1 == x2 and max(y1, y2) > 2:
                G[edge[0]][edge[1]]["D"] = 0.8
                G[edge[0]][edge[1]]["W"] = 0.3
            else:
                G[edge[0]][edge[1]]["D"] = 1
                G[edge[0]][edge[1]]["W"] = 0.3

    def modify_edge_size(self, G, row_noise, column_noise, col_compressible=False):
        for edge in G.edges():
            (x1, y1) = G.nodes[edge[0]]["coo"]
            (x2, y2) = G.nodes[edge[1]]["coo"]
            if y1 == y2:
                if y1 < self.transfer_row:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"] + G[edge[0]][edge[1]]["D"] * row_noise[0, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"] + G[edge[0]][edge[1]]["W"] * row_noise[0, 1]
                    )
                elif y1 == self.transfer_row:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"] + G[edge[0]][edge[1]]["D"] * row_noise[1, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"] + G[edge[0]][edge[1]]["W"] * row_noise[1, 1]
                    )
                else:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"] + G[edge[0]][edge[1]]["D"] * row_noise[2, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"] + G[edge[0]][edge[1]]["W"] * row_noise[2, 1]
                    )
            elif col_compressible:
                if x1 == x2 and min(y1, y2) >= self.transfer_row:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"] + G[edge[0]][edge[1]]["D"] * column_noise[0, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"] + G[edge[0]][edge[1]]["W"] * column_noise[0, 1]
                    )
                elif x1 == x2 and min(y1, y2) < self.transfer_row:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"] + G[edge[0]][edge[1]]["D"] * column_noise[1, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"] + G[edge[0]][edge[1]]["W"] * column_noise[1, 1]
                    )
            else:
                G[edge[0]][edge[1]]["D"] = 5
                G[edge[0]][edge[1]]["W"] = 5

        return G

    def set_coo(self, G):
        for node in G.nodes():
            G.nodes[node]["coo"] = node

    def set_DOF(self, G):
        for node in G.nodes():
            x, y = G.nodes[node]["coo"]
            if y == 0:
                G.nodes[node]["DOF"] = [0, 0, 0]
                G.nodes[node]["free"] = [0]
            else:
                G.nodes[node]["DOF"] = [1, 1, 1]
                G.nodes[node]["free"] = [1]

    def set_loads(self, G):
        for node in G.nodes():
            x, y = G.nodes[node]["pos"]
            if x == 0 and y > self.transfer_row:
                G.nodes[node]["load"] = [7 * 1e3 * (y - self.transfer_row), -20 * 1e3, 0]
            else:
                G.nodes[node]["load"] = [0, -20 * 1e3, 0]

    def set_rotation(self, G):
        for u, v in G.edges():
            pos_u = G.nodes[u]["pos"]
            pos_v = G.nodes[v]["pos"]
            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]
            rotation_angle = math.atan2(dy, dx)
            G[u][v]["rotation"] = rotation_angle

    def set_real_flag(self, G):
        for edge in G.edges():
            G[edge[0]][edge[1]]["real"] = True

    def add_hop_edges(self, G):
        for edge in G.edges():
            if "real" not in G[edge[0]][edge[1]]:
                G[edge[0]][edge[1]]["real"] = True

        for node_i in G.nodes():
            x1, y1 = G.nodes[node_i]["coo"]
            if y1 == self.transfer_row:
                for node_j in G.nodes():
                    x2, y2 = G.nodes[node_j]["coo"]
                    if y2 > self.transfer_row and x1 == x2 and G.degree(node_j) > 2:
                        G.add_edge(node_i, node_j, real=False)

    def set_foundation_flag(self, G):
        for u, v in G.edges():
            free_u = G.nodes[u]["free"]
            free_v = G.nodes[v]["free"]
            if free_u == [0] or free_v == [0]:
                G[u][v]["foundation"] = 1
            else:
                G[u][v]["foundation"] = 0

    def set_level(self, G):
        for u, v in G.edges():
            level_u = G.nodes[u]["coo"][1]
            level_v = G.nodes[v]["coo"][1]
            G[u][v]["level"] = (level_u + level_v) / 2
            G[u][v]["level_ratio"] = (max(level_u, level_v) - self.transfer_row) / (
                self.num_rows - self.transfer_row
            )

    def set_column_l2r_ratio(self, G):
        for u, v in G.edges():
            col_u = G.nodes[u]["coo"][0]
            col_v = G.nodes[v]["coo"][0]
            G[u][v]["col_l2r_ratio"] = (col_u + col_v) / self.num_cols

    def set_transfer_flag(self, G):
        for u, v in G.edges():
            x1, y1 = G.nodes[u]["coo"]
            x2, y2 = G.nodes[v]["coo"]
            if y1 == y2 and y1 == self.transfer_row:
                G[u][v]["transfer"] = 1
            else:
                G[u][v]["transfer"] = 0

    def set_cant_flag(self, G):
        for u, v in G.edges():
            if (not G[u][v]["column"]) and (np.minimum(G.degree[u], G.degree[v]) == 1):
                G[u][v]["cant"] = 1
            else:
                G[u][v]["cant"] = 0

    def set_step_size(self, G):
        for u, v in G.edges():
            G[u][v]["step_size"] = 0.05

    def set_column_flag(self, G):
        for u, v in G.edges():
            if np.abs(np.sin(G[u][v]["rotation"])) > 0.9:
                G[u][v]["column"] = True
            else:
                G[u][v]["column"] = False

    def set_distance(self, G):
        for u, v in G.edges():
            pos_u = G.nodes[u]["pos"]
            pos_v = G.nodes[v]["pos"]
            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]
            G[u][v]["dist"] = np.sqrt(dx**2 + dy**2)

    @staticmethod
    def node_tuple_2_index(G):
        mapping = {node: index for index, node in enumerate(G.nodes())}
        G_relabel = nx.relabel_nodes(G, mapping)
        return G_relabel

    def calc_col_distance(self, cols):
        cols = np.sort(cols)

        a = np.diff(cols, prepend=0)
        b = np.diff(cols, append=self.num_cols)
        max_diff = np.maximum(a, b).astype("float")

        a = np.diff(cols, prepend=-1000)
        b = np.diff(cols, append=1000)
        min_diff = np.minimum(a, b).astype("float")

        max_diff[0] = max_diff[0] + 3 / self.horizontal_scale
        max_diff[-1] = max_diff[-1] + 3 / self.horizontal_scale
        return max_diff * self.horizontal_scale, min_diff * self.horizontal_scale

    def sample_cols(self, cols):
        while True:
            max_dist, min_dist = self.calc_col_distance(cols)
            distances = np.sqrt(max_dist * min_dist)
            new_cols = np.sort(cols)
            while True:
                if min(min_dist) >= self.distance_lower_bound:
                    if np.random.rand() > (self.distance_lower_bound * 1.2) / min(min_dist):
                        break
                keep_prob = np.exp(distances) / sum(np.exp(distances))
                new_cols = np.sort(
                    np.random.choice(new_cols, size=len(new_cols) - 1, replace=False, p=keep_prob)
                )
                if len(new_cols) < 2:
                    break
                max_dist, min_dist = self.calc_col_distance(new_cols)
                distances = np.sqrt(max_dist * min_dist)
            if max(max_dist) < self.distance_upper_bound:
                break

        return new_cols

    def remove_cols(self, G, random_cols_up, random_cols_down):
        edges_to_remove = []
        for edge in G.edges():
            (x1, y1) = G.nodes[edge[0]]["coo"]
            (x2, y2) = G.nodes[edge[1]]["coo"]
            if y1 != y2:
                if (y1 >= self.transfer_row and x1 not in random_cols_up) or (
                    y1 < self.transfer_row and x1 not in random_cols_down
                ):
                    edges_to_remove.append(edge)

        G.remove_edges_from(edges_to_remove)
        return G

    def unify_edges(self, G):
        while True:
            degree_two_nodes = [
                node
                for node in G.nodes()
                if G.degree(node) == 2
                and G.nodes()[node]["coo"][0] != 0
                and G.nodes()[node]["coo"][0] != self.num_cols - 1
            ]
            if not degree_two_nodes:
                break

            for node in degree_two_nodes:
                neighbors = list(G.neighbors(node))
                if len(neighbors) == 2:
                    y, z = neighbors
                    if G.nodes()[y]["coo"][1] == G.nodes()[z]["coo"][1]:
                        G.remove_node(node)
                        if not G.has_edge(y, z):
                            G.add_edge(y, z)

        return G

    def remove_disconnected_nodes(self, G):
        nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 0]
        G.remove_nodes_from(nodes_to_remove)
        return G

    def remove_foundation_beams(self, G):
        edges_to_remove = []
        for edge in G.edges():
            (x1, y1) = G.nodes[edge[0]]["coo"]
            (x2, y2) = G.nodes[edge[1]]["coo"]
            if y1 == 0 and y2 == 0:
                edges_to_remove.append(edge)

        G.remove_edges_from(edges_to_remove)
        return G

    def set_d_theta(self, G, U):
        U = U.reshape(-1, 3)
        U = U[:, 2]
        for node in G.nodes():
            G.nodes[node]["d_theta"] = U[node]
        return G

    def agg_deflections(self, G, deflections):
        for i, edge in enumerate(G.edges()):
            d_max = np.abs(deflections[i]).max(axis=0)
            d_min = np.abs(deflections[i]).min(axis=0)
            G.edges[edge]["def_axial_max"] = d_max[0]
            G.edges[edge]["def_perp_max"] = d_max[1]
            G.edges[edge]["def_axial_min"] = d_min[0]
            G.edges[edge]["def_perp_min"] = d_min[1]
            G.edges[edge]["def_axial_diff"] = d_max[0] - d_min[0]
            G.edges[edge]["def_perp_diff"] = d_max[1] - d_min[1]
        return G

    def agg_deflection(self, G, Gs, deflections):
        for us, vs in Gs.edges():
            x1, y1 = Gs.nodes[us]["coo"]
            x2, y2 = Gs.nodes[vs]["coo"]
            Gs[us][vs]["deflection"] = np.zeros((0, 2))
            for i, (u, v) in enumerate(G.edges()):
                x3, y3 = G.nodes[u]["coo"]
                x4, y4 = G.nodes[v]["coo"]
                if np.abs(np.cos(Gs[us][vs]["rotation"])) > 0.9:  # slab
                    if (
                        max(x1, x3) < min(x2, x4)
                        and max(y1, y2) == max(y3, y4)
                        and min(y1, y2) == min(y3, y4)
                    ):
                        Gs[us][vs]["deflection"] = np.vstack((Gs[us][vs]["deflection"], deflections[i]))
                else:  # column
                    if (
                        max(y1, y2) == max(y3, y4)
                        and min(y1, y2) == min(y3, y4)
                        and (x1 == x2)
                        and (x2 == x3)
                        and (x3 == x4)
                    ):
                        Gs[us][vs]["deflection"] = deflections[i]
        return Gs

    def max_distance_to_line(self, x, y):
        slope = (y[-1] - y[0]) / (x[-1] - x[0])
        intercept = y[0] - slope * x[0]
        distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
        return np.max(distances)

    def max_distance_to_cant(self, x, y, slope_l, slop_r):
        slope = slope_l
        intercept = y[0] - slope * x[0]
        distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
        dist1 = np.max(distances)

        slope = slop_r
        intercept = y[-1] - slope * x[-1]
        distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
        dist2 = np.max(distances)

        return np.maximum(dist1, dist2)

    def calc_ver_deflection(self, Gs):
        for u, v in Gs.edges():
            if "valid" not in Gs[u][v]:
                Gs[u][v]["valid"] = True
            if "normal_deflection" not in Gs[u][v]:
                Gs[u][v]["normal_deflection"] = 0
            if np.abs(np.cos(Gs[u][v]["rotation"])) > 0.9:  # slab
                if Gs.nodes[u]["coo"][0] < Gs.nodes[v]["coo"][0]:
                    x = np.arange(0, Gs[u][v]["dist"], Gs[u][v]["step_size"])
                else:
                    x = np.arange(0, Gs[u][v]["dist"], -Gs[u][v]["step_size"])
                x = x + Gs[u][v]["deflection"][:, 0]
                y = Gs[u][v]["deflection"][:, 1]
                if Gs[u][v]["cant"]:
                    Gs[u][v]["normal_deflection"] = (
                        self.max_distance_to_cant(x, y, Gs.nodes[u]["d_theta"], Gs.nodes[v]["d_theta"])
                        / Gs[u][v]["dist"]
                    )
                else:
                    Gs[u][v]["normal_deflection"] = self.max_distance_to_line(x, y) / Gs[u][v]["dist"]
                Gs[u][v]["valid"] = Gs[u][v]["normal_deflection"] < 1 / 2e3

        return Gs

    def calc_ver_deflections(self, G, Gs, deflections):
        G = self.agg_deflection(G, Gs, deflections)
        for us, vs in Gs.edges():
            x1, y1 = Gs.nodes[us]["coo"]
            x2, y2 = Gs.nodes[vs]["coo"]
            Gs[us][vs]["def_perp_max"] = -100
            Gs[us][vs]["def_perp_min"] = 100
            column_compress = -100

            for u, v in G.edges():
                x3, y3 = G.nodes[u]["coo"]
                x4, y4 = G.nodes[v]["coo"]

                if (
                    max(y1, y2) == max(y3, y4)
                    and min(y1, y2) == min(y3, y4)
                    and max(x1, x2) == max(x3, x4)
                    and min(x1, x2) == min(x3, x4)
                ) or (
                    max(x1, x3) < min(x2, x4) and max(y1, y2) == max(y3, y4) and min(y1, y2) == min(y3, y4)
                ):
                    Gs[us][vs]["def_perp_max"] = max(Gs[us][vs]["def_perp_max"], G[u][v]["def_perp_max"])
                    Gs[us][vs]["def_perp_min"] = min(Gs[us][vs]["def_perp_min"], G[u][v]["def_perp_min"])

                    if (min(x3, x4) == min(x1, x2)) or (max(x3, x4) == max(x1, x2)):
                        column_compress = max(column_compress, G[u][v]["def_perp_min"])

            Gs[us][vs]["def_perp_diff"] = Gs[us][vs]["def_perp_max"] - Gs[us][vs]["def_perp_min"]
            Gs[us][vs]["valid"] = int(Gs[us][vs]["def_perp_diff"] < Gs[us][vs]["dist"] / 2e3)
        return Gs

    def calc_drift(self, G, Gs, U):
        U = U.reshape(-1, 3)
        U = U[:, 0]

        for us, vs in Gs.edges():
            if "valid" not in Gs[us][vs]:
                Gs[us][vs]["valid"] = True
            if "drift" not in Gs[us][vs]:
                Gs[us][vs]["drift"] = 0
            if np.cos(Gs[us][vs]["rotation"]) < 0.1:
                x1, y1 = Gs.nodes[us]["coo"]
                x2, y2 = Gs.nodes[vs]["coo"]
                for u, v in G.edges():
                    x3, y3 = G.nodes[u]["coo"]
                    x4, y4 = G.nodes[v]["coo"]

                    if (
                        max(y1, y2) == max(y3, y4)
                        and min(y1, y2) == min(y3, y4)
                        and max(x1, x2) == max(x3, x4)
                        and min(x1, x2) == min(x3, x4)
                    ):
                        Gs[us][vs]["drift"] = np.abs(U[u] - U[v]) / Gs[us][vs]["dist"]
                        Gs[us][vs]["valid"] = Gs[us][vs]["drift"] < 1 / 5e2
        return Gs

    def copy_displacement(self, G, Gs):
        for node_Gs in Gs.nodes():
            Gs.nodes[node_Gs]["disp"] = [
                G.nodes[node_G]["disp"]
                for node_G in G.nodes()
                if Gs.nodes[node_Gs]["coo"] == G.nodes[node_G]["coo"]
            ][0]
        return Gs

    def visualize(self, G):
        pos = nx.get_node_attributes(G, "pos")
        fig = plt.figure(figsize=(5, 5))
        nx.draw(G, pos, with_labels=True, node_size=5, node_color="lightblue", font_size=0)
        plt.title("Multi-story frame")
        plt.savefig("G.png", bbox_inches="tight")

    @staticmethod
    def graph_to_array(G):
        dof_array = np.array([G.nodes[node]["DOF"] for node in G.nodes()])
        restrained_dof = np.where(dof_array.reshape(-1, 1) == 0)[0]

        edges = np.array([(u, v) for u, v in G.edges()])
        rotations = np.array([G[u][v]["rotation"] for u, v in G.edges()])
        lenghts = np.array([G[u][v]["dist"] for u, v in G.edges()])
        depths = np.array([G[u][v]["D"] for u, v in G.edges()])
        widths = np.array([G[u][v]["W"] for u, v in G.edges()])
        positions = np.array([G.nodes[node]["pos"] for node in G.nodes()])
        loads = np.array([G.nodes[node]["load"] for node in G.nodes()])
        loads = loads.reshape(-1, 1).ravel()
        step_sizes = np.array([G[u][v]["step_size"] for u, v in G.edges()])

        return (positions, restrained_dof, loads, edges, rotations, lenghts, depths, widths, step_sizes)

    def save_graph(self, G, directory):
        file_name = G.graph["name"]
        file_name = f"{file_name}.pkl"
        file_path = os.path.join(directory, file_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "wb") as file:
            pickle.dump(G, file)

    @staticmethod
    def save_pyg_line_graphs(nx_graphs, dir, name, has_label=True, use_virtual_node=True):
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, name)

        pyg_data_list = []
        for G in nx_graphs:
            G = G.to_directed()
            edge_index = torch.tensor(list(G.edges())).T
            G = nx.node_link_data(G)
            nodes = G["nodes"]

            x = torch.tensor(
                [
                    [
                        node["rotation"] > 0.01,
                        node["cant"],
                        node["foundation"],
                        node["transfer"],
                        node.get("real", 1),
                        node["level_ratio"],
                        node["col_l2r_ratio"] * np.sin(node["rotation"]),
                        node["level"],
                        node["D"] ** 3 * node["W"] * np.sin(node["rotation"]),
                        node["D"] * node["W"] * np.sin(node["rotation"]),
                        node["D"] ** 3 * node["W"] * np.cos(node["rotation"]),
                        node["D"] * node["W"] * np.cos(node["rotation"]),
                        node["dist"] * np.cos(node["rotation"]),
                    ]
                    for node in nodes
                ],
                dtype=torch.float,
            )
            weight = sum([node["D"] * node["W"] * node["dist"] * int(node.get("real", 1)) for node in nodes])

            # Laplacian Positional Encoding (computed on line graph before virtual node)
            pe = _laplacian_pe(edge_index, num_nodes=x.shape[0], k=8)
            x = torch.cat([x, pe], dim=1)  # [N, 13] → [N, 21]

            if use_virtual_node:
                # Add virtual node (all-zero features; real=0 at index 4 marks it as non-physical)
                num_real_nodes = x.shape[0]
                vn_feat = torch.zeros(1, x.shape[1])
                x = torch.cat([x, vn_feat], dim=0)
                vn_idx = num_real_nodes
                real_indices = torch.arange(num_real_nodes, dtype=torch.long)
                vn_to_all = torch.stack([torch.full((num_real_nodes,), vn_idx, dtype=torch.long), real_indices])
                all_to_vn = torch.stack([real_indices, torch.full((num_real_nodes,), vn_idx, dtype=torch.long)])
                edge_index = torch.cat([edge_index, vn_to_all, all_to_vn], dim=1)

            if has_label:
                y = torch.tensor([node["valid"] for node in nodes])
                drift = torch.tensor([node["drift"] for node in nodes])
                normal_def = torch.tensor([node["normal_deflection"] for node in nodes])
                if use_virtual_node:
                    # Append dummy label for virtual node (masked out during training)
                    y = torch.cat([y, torch.zeros(1, dtype=y.dtype)])
                    drift = torch.cat([drift, torch.zeros(1, dtype=drift.dtype)])
                    normal_def = torch.cat([normal_def, torch.zeros(1, dtype=normal_def.dtype)])
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    weight=weight,
                    drift=drift,
                    normal_def=normal_def,
                    name=G["graph"]["name"],
                )
            else:
                data = Data(x=x, edge_index=edge_index, weight=weight, name=G["graph"]["name"])

            pyg_data_list.append(data)

        with open(path, "wb") as f:
            pickle.dump(pyg_data_list, f)

    @staticmethod
    def load_all_graphs(directory):
        graphs = []
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                file_path = os.path.join(directory, filename)
                with open(file_path, "rb") as file:
                    graph = pickle.load(file)
                    if isinstance(graph, nx.Graph):
                        graphs.append(graph)
        return graphs

    @staticmethod
    def load_graphs(directory, file_names):
        graphs = []
        for name in file_names:
            file_path = os.path.join(directory, name)
            with open(file_path, "rb") as file:
                graph = pickle.load(file)
                if isinstance(graph, nx.Graph):
                    graphs.append(graph)
        return graphs

    @staticmethod
    def draw_graph(Gs, path):
        from fea_gnn_surrogate.visualization import plot_deflection

        (
            node_positions,
            node_restrained_dof,
            node_loads,
            edges,
            edge_rotations,
            edge_lenghts,
            edge_depths,
            edge_widths,
            step_sizes,
        ) = GraphHandler.graph_to_array(Gs)
        name = Gs.graph["name"]
        valid_flags = [Gs[u][v]["valid"] for u, v in Gs.edges()]
        deflections = [Gs[u][v]["deflection"] for u, v in Gs.edges()]
        plot_deflection(
            edges,
            node_positions,
            edge_rotations,
            edge_lenghts,
            deflections,
            edge_depths,
            edge_widths,
            100,
            path,
            name,
            valid_flags,
        )
