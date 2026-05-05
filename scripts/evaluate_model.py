"""Report test set metrics for a trained GNN surrogate model."""
import argparse
import os
import torch
from torch_geometric.loader import DataLoader

from fea_gnn_surrogate.surrogate.inference import load_model
from fea_gnn_surrogate.surrogate.dataset import load_dataset, normalize_data
from fea_gnn_surrogate.surrogate.train import validate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GNN surrogate on one or more test datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/evaluate_model.py \\
      --model_path best_model.pth \\
      --test_dataset_paths \\
          data/test_1/with_vn/dataset/pyg_line_graphs.pkl \\
          data/test_2/with_vn/dataset/pyg_line_graphs.pkl
""",
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--test_dataset_paths", type=str, nargs="+", required=True,
                        help="One or more paths to test PyG datasets (.pkl)")
    parser.add_argument("--hidden_dim", type=int, default=18,
                        help="Hidden dimension, must match training (default: 18)")
    parser.add_argument("--mp_steps", type=int, default=3,
                        help="Message passing steps, must match training (default: 3)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation (default: 32)")
    args = parser.parse_args()

    # Load the first test set to infer num_features
    first_data = load_dataset(args.test_dataset_paths[0])
    num_features = first_data[0].x.shape[1]

    model, norm_stats = load_model(
        args.model_path, num_features,
        hidden_dim=args.hidden_dim, num_mp_steps=args.mp_steps,
    )

    # pos_weight=1 during eval so loss is unweighted BCE (comparable across datasets)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Model:  {args.model_path}")
    print(f"{'Dataset':<50}  {'Loss':>8}  {'AUC':>8}")
    print("-" * 70)

    for test_path in args.test_dataset_paths:
        test_data = load_dataset(test_path)
        if norm_stats is not None:
            normalize_data(test_data, norm_stats)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        loss, auc, _, _, _ = validate(model, test_loader, criterion)
        label = os.path.basename(os.path.dirname(os.path.dirname(test_path)))
        print(f"  {label:<48}  {loss:>8.4f}  {auc:>8.4f}")


if __name__ == "__main__":
    main()
