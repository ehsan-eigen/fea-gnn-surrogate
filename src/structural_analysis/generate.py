import os
import numpy as np
import networkx as nx

from structural_analysis.config import load_config
from structural_analysis.fea.environment import StructEnvironment
from structural_analysis.fea import nodal_reactions as nr
from structural_analysis.graph.graph_utils import GraphHandler


def generate_samples(config_path, mode="train", num_episodes=1000,
                     fea=True, visualize=False, image_dir="images/",
                     save_graphs=False, output_dir="data"):
    """Generate structural samples and run FEA.

    Returns a list of line graphs (PyG-ready NetworkX graphs).
    """
    conf = load_config(config_path, mode)
    env = StructEnvironment()
    graph_handler = GraphHandler(conf)

    episode = 0
    line_graphs = []

    while episode < num_episodes:
        if mode == "train":
            graph_handler.num_rows = conf["num_rows"] + np.random.randint(low=-3, high=3)

        G = graph_handler.generate_graph(mode=mode)
        Gs = graph_handler.simplify_graph(G)

        row_noise = np.random.beta(4, 4, size=(3, 2))
        column_noise = np.random.beta(4, 4, size=(2, 2))

        G = graph_handler.modify_edge_size(G, row_noise, column_noise, col_compressible=True)
        Gs = graph_handler.modify_edge_size(Gs, row_noise, column_noise, col_compressible=True)

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
        ) = GraphHandler.graph_to_array(G)

        env.set_attributes(
            node_positions,
            node_restrained_dof,
            node_loads,
            edges,
            edge_rotations,
            edge_lenghts,
            edge_depths,
            edge_widths,
            step_sizes,
        )

        if fea:
            UG, deflections, K = env.analyse()
            if UG is None or UG.max() > 0.2:
                print("invalid")
                continue
            Gs = graph_handler.agg_deflection(G, Gs, deflections)
            Gs = graph_handler.set_d_theta(Gs, UG)
            Gs = graph_handler.calc_ver_deflection(Gs)
            Gs = graph_handler.calc_drift(G, Gs, UG)

        L = nx.line_graph(Gs)
        L.add_nodes_from((node, Gs.edges[node]) for node in L)
        L = GraphHandler.node_tuple_2_index(L)

        G.graph["name"] = str(episode)
        Gs.graph["name"] = str(episode)
        L.graph["name"] = str(episode)
        line_graphs.append(L)

        if save_graphs:
            graph_handler.save_graph(G, os.path.join(output_dir, mode, "raw", "graphs"))
            graph_handler.save_graph(Gs, os.path.join(output_dir, mode, "simplified", "graphs"))

        Gs = GraphHandler.node_tuple_2_index(Gs)
        if visualize:
            GraphHandler.draw_graph(Gs, image_dir)

        print(f"step: {episode}")
        episode += 1

    return line_graphs
