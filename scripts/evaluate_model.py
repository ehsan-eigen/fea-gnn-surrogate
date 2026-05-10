"""Report test set metrics for a trained GNN surrogate model."""
import argparse
import os
import torch
from torch_geometric.loader import DataLoader

from fea_gnn_surrogate.surrogate.inference import load_model
from fea_gnn_surrogate.surrogate.dataset import load_dataset, normalize_data
from fea_gnn_surrogate.surrogate.train import validate


EDGE_STRATEGIES = ["with_vn", "hop_edges", "no_vn"]


def _dataset_path(data_dir, test_set, edge_strategy):
    return os.path.join(data_dir, test_set, edge_strategy, "dataset", "pyg_line_graphs.pkl")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GNN surrogate on one or more test datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/evaluate_model.py --edge_strategy with_vn --test_sets test_1 test_2
""",
    )
    parser.add_argument("--test_sets", type=str, nargs="+", required=True,
                        help="One or more test set names (e.g. test_1 test_2)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint (default: best_model_<edge_strategy>.pth)")
    parser.add_argument("--edge_strategy", type=str, default="with_vn",
                        choices=EDGE_STRATEGIES,
                        help="Edge strategy variant (default: with_vn)")
    parser.add_argument("--data_dir", type=str, default="hf://ehsan94/fea-gnn-surrogate",
                        help="Root data directory or hf://owner/repo (default: hf://ehsan94/fea-gnn-surrogate)")
    parser.add_argument("--hidden_dim", type=int, default=18,
                        help="Hidden dimension, must match training (default: 18)")
    parser.add_argument("--mp_steps", type=int, default=3,
                        help="Message passing steps / GPS layers, must match training (default: 3)")
    parser.add_argument("--heads", type=int, default=3,
                        help="Attention heads (GPS only, default: 3)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout (GPS only, default: 0.2)")
    parser.add_argument("--attn_dropout", type=float, default=0.2,
                        help="Attention dropout (GPS only, default: 0.2)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation (default: 32)")
    args = parser.parse_args()

    model_path = args.model_path or f"best_model_{args.edge_strategy}.pth"

    # Load the first test set to infer num_features
    first_path = _dataset_path(args.data_dir, args.test_sets[0], args.edge_strategy)
    first_data = load_dataset(first_path)
    num_features = first_data[0].x.shape[1]

    model, norm_stats = load_model(
        model_path, num_features,
        hidden_dim=args.hidden_dim, num_mp_steps=args.mp_steps,
        heads=args.heads, dropout=args.dropout, attn_dropout=args.attn_dropout,
    )

    # pos_weight=1 during eval so loss is unweighted BCE (comparable across datasets)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Model:  {model_path}")
    print(f"{'Test Set':<20}  {'Loss':>8}  {'AUC':>8}")
    print("-" * 40)

    for test_set in args.test_sets:
        test_path = _dataset_path(args.data_dir, test_set, args.edge_strategy)
        test_data = load_dataset(test_path)
        if norm_stats is not None:
            normalize_data(test_data, norm_stats)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        loss, auc, _, _, _ = validate(model, test_loader, criterion)
        print(f"  {test_set:<18}  {loss:>8.4f}  {auc:>8.4f}")


if __name__ == "__main__":
    main()
