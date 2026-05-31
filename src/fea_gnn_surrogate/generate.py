import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx

from fea_gnn_surrogate.config import load_configs, DEFAULT_TRAIN_DATASETS
from fea_gnn_surrogate.fea.environment import StructEnvironment
from fea_gnn_surrogate.graph.graph_utils import GraphHandler


def _generate_single_sample(configs, fea, save_graphs, output_dir, mode,
                             visualize, image_dir, episode_id):
    """Generate one structural sample. Returns (episode_id, L, G, Gs) or None if invalid."""
    conf = configs[np.random.randint(len(configs))]
    graph_handler = GraphHandler(conf)

    tr_dist = conf["transfer_row_dist"]
    tr_values = [int(k) for k in tr_dist.keys()]
    tr_probs = np.array(list(tr_dist.values()), dtype=float)
    tr_probs /= tr_probs.sum()
    graph_handler.transfer_row = int(np.random.choice(tr_values, p=tr_probs))

    nr_dist = conf["num_rows_dist"]
    nr_values = [int(k) for k in nr_dist.keys()]
    nr_probs = np.array(list(nr_dist.values()), dtype=float)
    nr_probs /= nr_probs.sum()
    graph_handler.num_rows = int(np.random.choice(nr_values, p=nr_probs))

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

    env = StructEnvironment()
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
            return None

        Gs = graph_handler.agg_deflection(G, Gs, deflections)
        Gs = graph_handler.set_d_theta(Gs, UG)
        Gs = graph_handler.calc_ver_deflection(Gs)
        Gs = graph_handler.calc_drift(G, Gs, UG)

    graph_handler.set_load_decomposition(Gs)
    L = nx.line_graph(Gs)
    L.add_nodes_from((node, Gs.edges[node]) for node in L)
    # Preserve joint-graph plumbing on the line graph for the physics-aware
    # attention bias. Each line-graph node (= a member) carries (u, v) integer
    # joint endpoints, and the line graph itself remembers how many joints
    # exist and which are supports. Gs is int-keyed but sparse (gaps left by
    # `remove_disconnected_nodes` / `unify_edges`), so we densify joint IDs
    # to [0, num_joints) here — must match the densification applied to Gs
    # at the end of this function.
    gs_to_dense = {node: i for i, node in enumerate(Gs.nodes())}
    for node in L.nodes():
        u, v = node
        L.nodes[node]["endpoints"] = (gs_to_dense[u], gs_to_dense[v])
    L.graph["num_joints"] = Gs.number_of_nodes()
    L.graph["joint_supports"] = sorted(
        gs_to_dense[j] for j in Gs.nodes() if Gs.nodes[j].get("free") == [0]
    )
    L = GraphHandler.node_tuple_2_index(L)

    G.graph["name"] = str(episode_id)
    Gs.graph["name"] = str(episode_id)
    L.graph["name"] = str(episode_id)

    if save_graphs:
        graph_handler.save_graph(G, os.path.join(output_dir, mode, "raw", "graphs"))
        graph_handler.save_graph(Gs, os.path.join(output_dir, mode, "simplified", "graphs"))

    Gs = GraphHandler.node_tuple_2_index(Gs)
    if visualize:
        GraphHandler.draw_graph(Gs, image_dir)

    return (episode_id, L)


def generate_samples(config_path, datasets=None, num_episodes=1000,
                     mode="train", fea=True,
                     visualize=False, image_dir="images/",
                     save_graphs=False, output_dir="data",
                     concurrency=1):
    """Generate structural samples and run FEA.

    Parameters
    ----------
    config_path : str
        Path to config.json.
    datasets : list[str] or str, optional
        Dataset key(s) in config.json. Each episode randomly picks one
        config from the list. Defaults to DEFAULT_TRAIN_DATASETS.
    mode : str
        "train" or "test". Used for output directory labelling.
    concurrency : int
        Number of samples to generate in parallel (default: 1 = sequential).
    """
    if datasets is None:
        datasets = DEFAULT_TRAIN_DATASETS

    if isinstance(datasets, str):
        datasets = [datasets]

    configs = load_configs(config_path, datasets)

    if concurrency <= 1:
        return _generate_sequential(configs, num_episodes, mode, fea,
                                    visualize, image_dir, save_graphs, output_dir)

    return _generate_parallel(configs, num_episodes, mode, fea,
                              visualize, image_dir, save_graphs, output_dir,
                              concurrency)


def _generate_sequential(configs, num_episodes, mode, fea,
                         visualize, image_dir, save_graphs, output_dir):
    """Original sequential generation loop."""
    episode = 0
    line_graphs = []

    while episode < num_episodes:
        result = _generate_single_sample(
            configs, fea, save_graphs, output_dir, mode,
            visualize, image_dir, episode)

        if result is None:
            print("invalid")
            continue

        _, L = result
        line_graphs.append(L)
        print(f"step: {episode}")
        episode += 1

    return line_graphs


def _generate_parallel(configs, num_episodes, mode, fea,
                       visualize, image_dir, save_graphs, output_dir,
                       concurrency):
    """Parallel generation using a process pool."""
    line_graphs = []
    episode = 0
    next_id = 0

    with ProcessPoolExecutor(max_workers=concurrency) as executor:
        futures = {}

        # Seed initial batch of tasks
        for _ in range(min(concurrency, num_episodes)):
            fut = executor.submit(
                _generate_single_sample,
                configs, fea, save_graphs, output_dir, mode,
                visualize, image_dir, next_id)
            futures[fut] = next_id
            next_id += 1

        while futures:
            for fut in as_completed(futures):
                submitted_id = futures.pop(fut)
                result = fut.result()

                if result is None:
                    print("invalid")
                    # Resubmit with the same episode id
                    new_fut = executor.submit(
                        _generate_single_sample,
                        configs, fea, save_graphs, output_dir, mode,
                        visualize, image_dir, submitted_id)
                    futures[new_fut] = submitted_id
                else:
                    _, L = result
                    line_graphs.append(L)
                    episode += 1
                    print(f"step: {submitted_id} ({episode}/{num_episodes})")

                    # Submit next task if needed
                    if next_id < num_episodes:
                        new_fut = executor.submit(
                            _generate_single_sample,
                            configs, fea, save_graphs, output_dir, mode,
                            visualize, image_dir, next_id)
                        futures[new_fut] = next_id
                        next_id += 1

                # Break inner loop to refresh as_completed with updated futures
                break

    # Sort by episode id for deterministic ordering
    line_graphs.sort(key=lambda g: int(g.graph["name"]))
    return line_graphs
