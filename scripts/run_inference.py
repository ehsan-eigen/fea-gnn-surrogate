"""Run inference with a trained GNN surrogate model on test configurations."""
import argparse
import os
import pickle

from fea_gnn_surrogate.config import load_config
from fea_gnn_surrogate.fea.environment import StructEnvironment
from fea_gnn_surrogate.graph.graph_utils import GraphHandler
from fea_gnn_surrogate.surrogate.inference import load_model, predict, rank_structures
from fea_gnn_surrogate.surrogate.dataset import normalize_data


EDGE_STRATEGIES = ["with_vn", "no_vn"]


def main():
    parser = argparse.ArgumentParser(description="Run GNN surrogate inference on test structures")
    parser.add_argument("--test_name", type=str, required=True,
                        help="Config section for the test case (e.g. dataset_2, dataset_3, ...)")
    parser.add_argument("--edge_strategy", type=str, default="with_vn",
                        choices=EDGE_STRATEGIES,
                        help="Edge strategy variant (default: with_vn)")
    parser.add_argument("--model", type=str, default="mpnn", choices=["mpnn", "gps"],
                        help="Model architecture: mpnn or gps (default: mpnn)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model checkpoint (default: best_model_<model>_<edge_strategy>.pth)")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the configuration file (default: config.json)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory, must match --output_dir used during generation (default: data)")
    parser.add_argument("--top_stability", type=int, default=15,
                        help="Keep top-K structures by predicted validity score (default: 15)")
    parser.add_argument("--top_weight", type=int, default=15,
                        help="From those, keep the top-K lightest structures (default: 15)")
    parser.add_argument("--output_dir", type=str, default="top_structs",
                        help="Directory for deflection plot output (default: top_structs)")
    parser.add_argument("--hidden_dim", type=int, default=18,
                        help="Hidden dimension, must match training (default: 18)")
    parser.add_argument("--mp_steps", type=int, default=3,
                        help="Message passing steps, must match training (default: 3)")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"best_model_{args.model}_{args.edge_strategy}.pth"

    dataset_path = os.path.join(
        args.data_dir, args.test_name, args.edge_strategy, "dataset", "pyg_line_graphs.pkl"
    )

    with open(dataset_path, "rb") as f:
        val_data = pickle.load(f)

    num_features = val_data[0].x.shape[1]
    model, norm_stats = load_model(
        args.model_path, num_features,
        hidden_dim=args.hidden_dim, num_mp_steps=args.mp_steps,
        model_type=args.model,
    )

    if norm_stats is not None:
        normalize_data(val_data, norm_stats)

    results = predict(model, val_data)
    df = rank_structures(results, top_stability_count=args.top_stability, top_weight_count=args.top_weight)
    print(df.to_string())

    # Run FEA on top structures and visualize
    conf = load_config(args.config, args.test_name)
    graph_handler = GraphHandler(conf)

    top_graph_indices = df["name"].values.astype("int")
    top_graph_names = [str(idx) + ".pkl" for idx in top_graph_indices]

    graph_path = os.path.join(args.data_dir, args.test_name, "simplified", "graphs")
    top_simplified_graphs = GraphHandler.load_graphs(graph_path, top_graph_names)

    raw_graph_path = os.path.join(args.data_dir, args.test_name, "raw", "graphs")
    top_raw_graphs = GraphHandler.load_graphs(raw_graph_path, top_graph_names)

    env = StructEnvironment()
    output_path = os.path.join(args.output_dir, args.test_name)

    for i in range(len(top_raw_graphs)):
        G = top_raw_graphs[i]
        Gs = top_simplified_graphs[i]

        (
            node_positions, node_restrained_dof, node_loads, edges,
            edge_rotations, edge_lenghts, edge_depths, edge_widths, step_sizes,
        ) = GraphHandler.graph_to_array(G)

        env.set_attributes(
            node_positions, node_restrained_dof, node_loads, edges,
            edge_rotations, edge_lenghts, edge_depths, edge_widths, step_sizes,
        )
        UG, deflections, K = env.analyse()
        Gs = graph_handler.agg_deflection(G, Gs, deflections)
        Gs = graph_handler.set_d_theta(Gs, UG)
        Gs = graph_handler.calc_ver_deflection(Gs)
        Gs = graph_handler.calc_drift(G, Gs, UG)
        Gs = GraphHandler.node_tuple_2_index(Gs)
        GraphHandler.draw_graph(Gs, output_path)

        weight = df["weight"].iloc[i]
        name = df["name"].iloc[i]
        print(f"name: {int(name)} weight: {int(weight)}")


if __name__ == "__main__":
    main()
