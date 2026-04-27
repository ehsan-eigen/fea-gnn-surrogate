"""Generate structural samples, run FEA, and save as PyG line graphs."""
import argparse
import os

from structural_analysis.generate import generate_samples
from structural_analysis.graph.graph_utils import GraphHandler


def main():
    parser = argparse.ArgumentParser(
        description="Generate random frame structures, run FEA to label elements, "
                    "and save as PyG graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Generate 1000 training samples
  python scripts/generate_dataset.py --mode train --num_samples 1000

  # Generate 500 test samples, also saving raw graphs for later visualization
  python scripts/generate_dataset.py --mode test_1 --num_samples 500 --save_graphs

  # Generate without running FEA (no validity labels)
  python scripts/generate_dataset.py --mode train --skip_fea
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
    parser.add_argument("--save_graphs", action="store_true",
                        help="Also save raw and simplified NetworkX graphs to disk. "
                             "Required if you want to visualize top structures during inference (Stage 3)")
    args = parser.parse_args()

    line_graphs = generate_samples(
        config_path=args.config,
        mode=args.mode,
        num_episodes=args.num_samples,
        fea=not args.skip_fea,
        visualize=args.visualize,
        save_graphs=args.save_graphs,
        output_dir=args.output_dir,
    )

    save_dir = os.path.join(args.output_dir, args.mode, "dataset")
    has_label = not args.skip_fea
    GraphHandler.save_pyg_line_graphs(line_graphs, save_dir, args.output_name, has_label=has_label)

    print(f"Saved {len(line_graphs)} graphs to {os.path.join(save_dir, args.output_name)}")


if __name__ == "__main__":
    main()
