"""Generate structural samples, run FEA, and save as PyG line graphs."""
import argparse
import os

from fea_gnn_surrogate.generate import generate_samples
from fea_gnn_surrogate.graph.graph_utils import GraphHandler


def _upload_to_hf(local_file, path_in_repo, repo_id):
    from huggingface_hub import upload_file
    upload_file(
        path_or_fileobj=local_file,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  Uploaded to hf://{repo_id}/{path_in_repo}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random frame structures, run FEA to label elements, "
                    "and save as PyG graphs for all three edge-strategy variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/generate_dataset.py --mode train --num_samples 1000
  python scripts/generate_dataset.py --mode test_1 --num_samples 500
""",
    )
    parser.add_argument("--mode", type=str, default="train",
                        help="Config section to use from config.json (default: train). "
                             "Available sections: train, test_1, test_2, test_3, test_4, test_5")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of structures to generate (default: 1000)")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the configuration file (default: config.json)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Root output directory (default: data)")
    parser.add_argument("--output_name", type=str, default="pyg_line_graphs.pkl",
                        help="Filename for the saved PyG dataset (default: pyg_line_graphs.pkl)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a deflection plot (PNG) for each generated structure")
    parser.add_argument("--skip_fea", action="store_true",
                        help="Skip finite element analysis. Structures will have no validity labels, "
                             "so they cannot be used for training — only for unlabelled inference")
    parser.add_argument("--no_save_graphs", action="store_true",
                        help="Do not save raw and simplified NetworkX graphs to disk "
                             "(they are saved by default for visualization in Stage 3)")
    parser.add_argument("--hf_repo", type=str, default="ehsan94/fea-gnn-surrogate",
                        help="Hugging Face dataset repo to upload datasets to "
                             "(default: ehsan94/fea-gnn-surrogate). Reads HF_TOKEN from environment.")
    args = parser.parse_args()

    save_graphs = not args.no_save_graphs

    line_graphs, line_graphs_hop = generate_samples(
        config_path=args.config,
        mode=args.mode,
        num_episodes=args.num_samples,
        fea=not args.skip_fea,
        visualize=args.visualize,
        save_graphs=save_graphs,
        output_dir=args.output_dir,
    )

    has_label = not args.skip_fea

    # Save base NX line graphs (topology + attributes, no feature extraction).
    # Re-run only extract_features.py when changing features — no need to redo FEA.
    base_dir = os.path.join(args.output_dir, args.mode, "base")
    GraphHandler.save_base_line_graphs(line_graphs, base_dir, "line_graphs.pkl")
    GraphHandler.save_base_line_graphs(line_graphs_hop, base_dir, "line_graphs_hop.pkl")
    print(f"Saved base line graphs to {base_dir}/")

    variants = [
        ("hop_edges", line_graphs_hop, False),
        ("with_vn", line_graphs, True),
        ("no_vn", line_graphs, False),
    ]

    for variant_name, graphs, use_vn in variants:
        save_dir = os.path.join(args.output_dir, args.mode, variant_name, "dataset")
        GraphHandler.save_pyg_line_graphs(graphs, save_dir, args.output_name,
                                          has_label=has_label, use_virtual_node=use_vn)
        local_file = os.path.join(save_dir, args.output_name)
        print(f"Saved {len(graphs)} graphs to {local_file}")
        if args.hf_repo:
            path_in_repo = f"{args.mode}/{variant_name}/dataset/{args.output_name}"
            _upload_to_hf(local_file, path_in_repo, args.hf_repo)


if __name__ == "__main__":
    main()
