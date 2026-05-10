"""Extract PyG features from base NX line graphs.

Use this after changing features in save_pyg_line_graphs — no need to
regenerate structures or re-run FEA.

Examples:
  python scripts/extract_features.py --mode train
  python scripts/extract_features.py --mode test_1 --skip_fea
"""
import argparse
import os

from fea_gnn_surrogate.graph.graph_utils import GraphHandler


def main():
    parser = argparse.ArgumentParser(description="Extract PyG features from base NX line graphs")
    parser.add_argument("--mode", type=str, default="train",
                        help="Config section (default: train)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory (default: data)")
    parser.add_argument("--output_name", type=str, default="pyg_line_graphs.pkl",
                        help="Filename for the saved PyG dataset (default: pyg_line_graphs.pkl)")
    parser.add_argument("--skip_fea", action="store_true",
                        help="If set, base graphs have no labels (unlabelled inference only)")
    args = parser.parse_args()

    base_dir = os.path.join(args.data_dir, args.mode, "base")
    line_graphs = GraphHandler.load_base_line_graphs(os.path.join(base_dir, "line_graphs.pkl"))
    print(f"Loaded {len(line_graphs)} base line graphs from {base_dir}/")

    has_label = not args.skip_fea

    variants = [
        ("with_vn", line_graphs, True),
        ("no_vn", line_graphs, False),
    ]

    for variant_name, graphs, use_vn in variants:
        save_dir = os.path.join(args.data_dir, args.mode, variant_name, "dataset")
        GraphHandler.save_pyg_line_graphs(graphs, save_dir, args.output_name,
                                          has_label=has_label, use_virtual_node=use_vn)
        local_file = os.path.join(save_dir, args.output_name)
        print(f"Saved {len(graphs)} graphs to {local_file}")


if __name__ == "__main__":
    main()
