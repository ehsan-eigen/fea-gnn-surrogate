"""Generate structural samples, run FEA, and save as PyG line graphs."""
import argparse
import os

from fea_gnn_surrogate.config import DEFAULT_TRAIN_DATASETS
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
                    "and save as PyG graphs for both edge-strategy variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Training data (uses default training datasets):
  python scripts/generate_dataset.py --num_samples 1000

  # Training data from specific datasets:
  python scripts/generate_dataset.py --datasets dataset_1 dataset_4 --num_samples 1000

  # Test data:
  python scripts/generate_dataset.py --mode test --datasets dataset_2 dataset_3 --num_samples 500
""",
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Generation mode: train or test (default: train)")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Dataset keys from config.json (default: %(default)s). "
                             f"Available: dataset_1 through dataset_6. "
                             f"Training default: {DEFAULT_TRAIN_DATASETS}")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of structures to generate (default: 1000)")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the configuration file (default: config.json)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Root output directory (default: data)")
    parser.add_argument("--output_label", type=str, default=None,
                        help="Subdirectory label under output_dir (default: value of --mode)")
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
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="Hugging Face dataset repo to upload datasets to. "
                             "If not set, no upload is performed. Reads HF_TOKEN from environment.")
    args = parser.parse_args()

    datasets = args.datasets
    if datasets is None:
        datasets = DEFAULT_TRAIN_DATASETS

    label = args.output_label if args.output_label is not None else args.mode

    save_graphs = not args.no_save_graphs

    line_graphs = generate_samples(
        config_path=args.config,
        datasets=datasets,
        num_episodes=args.num_samples,
        mode=args.mode,
        fea=not args.skip_fea,
        visualize=args.visualize,
        save_graphs=save_graphs,
        output_dir=args.output_dir,
    )

    has_label = not args.skip_fea

    # Save base NX line graphs (topology + attributes, no feature extraction).
    # Re-run only extract_features.py when changing features — no need to redo FEA.
    base_dir = os.path.join(args.output_dir, label, "base")
    GraphHandler.save_base_line_graphs(line_graphs, base_dir, "line_graphs.pkl")
    print(f"Saved base line graphs to {base_dir}/")

    variants = [
        ("with_vn", line_graphs, True),
        ("no_vn", line_graphs, False),
    ]

    for variant_name, graphs, use_vn in variants:
        save_dir = os.path.join(args.output_dir, label, variant_name, "dataset")
        GraphHandler.save_pyg_line_graphs(graphs, save_dir, args.output_name,
                                          has_label=has_label, use_virtual_node=use_vn)
        local_file = os.path.join(save_dir, args.output_name)
        print(f"Saved {len(graphs)} graphs to {local_file}")
        if args.hf_repo:
            path_in_repo = f"{label}/{variant_name}/dataset/{args.output_name}"
            _upload_to_hf(local_file, path_in_repo, args.hf_repo)


if __name__ == "__main__":
    main()
