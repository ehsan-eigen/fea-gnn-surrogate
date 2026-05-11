import os
import numpy as np
import networkx as nx

from fea_gnn_surrogate.config import load_configs, DEFAULT_TRAIN_DATASETS
from fea_gnn_surrogate.fea.environment import StructEnvironment
from fea_gnn_surrogate.graph.graph_utils import GraphHandler


def generate_samples(config_path, datasets=None, num_episodes=1000,
                     training=True, num_rows_range=None, fea=True,
                     visualize=False, image_dir="images/",
                     save_graphs=False, output_dir="data"):
    """Generate structural samples and run FEA.

    Parameters
    ----------
    config_path : str
        Path to config.json.
    datasets : list[str] or str, optional
        Dataset key(s) in config.json. For training, each episode randomly
        picks one config from the list. Defaults to DEFAULT_TRAIN_DATASETS
        when training=True, or must be a single string when training=False.
    training : bool
        When True, applies per-episode augmentation: variable transfer_row,
        num_rows, and load sampling.
    num_rows_range : list[int], optional
        [min, max] range for uniform num_rows sampling during training.
        Defaults to [config_num_rows - 3, config_num_rows + 3] per dataset.
    """
    if datasets is None:
        datasets = DEFAULT_TRAIN_DATASETS if training else ["dataset_1"]

    if isinstance(datasets, str):
        datasets = [datasets]

    configs = load_configs(config_path, datasets)
    env = StructEnvironment()

    episode = 0
    line_graphs = []
    mode_label = "train" if training else datasets[0]

    while episode < num_episodes:
        conf = configs[np.random.randint(len(configs))]
        graph_handler = GraphHandler(conf)

        if training:
            # Vary transfer_row: 2 most likely, 3 less, 4 least
            graph_handler.transfer_row = np.random.choice(
                [2, 3, 4], p=[0.6, 0.3, 0.1]
            )
            # Vary num_rows: uniform within range
            if num_rows_range is not None:
                lo, hi = num_rows_range
            else:
                lo = conf["num_rows"] - 3
                hi = conf["num_rows"] + 3
            graph_handler.num_rows = np.random.randint(lo, hi + 1)

        graph_handler.sample_load_params()
        G = graph_handler.generate_graph()
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

        graph_handler.set_load_decomposition(Gs)
        L = nx.line_graph(Gs)
        L.add_nodes_from((node, Gs.edges[node]) for node in L)
        L = GraphHandler.node_tuple_2_index(L)

        G.graph["name"] = str(episode)
        Gs.graph["name"] = str(episode)
        L.graph["name"] = str(episode)
        line_graphs.append(L)

        if save_graphs:
            graph_handler.save_graph(G, os.path.join(output_dir, mode_label, "raw", "graphs"))
            graph_handler.save_graph(Gs, os.path.join(output_dir, mode_label, "simplified", "graphs"))

        Gs = GraphHandler.node_tuple_2_index(Gs)
        if visualize:
            GraphHandler.draw_graph(Gs, image_dir)

        print(f"step: {episode}")
        episode += 1

    return line_graphs
