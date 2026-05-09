"""Train the GNN surrogate model."""
import argparse
import os
import torch
from torch_geometric.loader import DataLoader

from fea_gnn_surrogate.surrogate.model import SharedMPNN
from fea_gnn_surrogate.surrogate.dataset import (
    load_dataset,
    create_dataloaders,
    compute_class_weight,
    compute_normalization_stats,
    normalize_data,
)
from fea_gnn_surrogate.surrogate.train import train, validate


EDGE_STRATEGIES = ["with_vn", "hop_edges", "no_vn"]


def _dataset_path(data_dir, mode, edge_strategy):
    return os.path.join(data_dir, mode, edge_strategy, "dataset", "pyg_line_graphs.pkl")


def main():
    parser = argparse.ArgumentParser(description="Train GNN surrogate model")
    parser.add_argument("--mode", type=str, default="train",
                        help="Config section for training data (default: train)")
    parser.add_argument("--edge_strategy", type=str, default="with_vn",
                        choices=EDGE_STRATEGIES,
                        help="Edge strategy variant (default: with_vn)")
    parser.add_argument("--test_sets", type=str, nargs="*", default=None,
                        help="Test sets to evaluate after training (e.g. test_1 test_2). "
                             "Uses the same edge strategy as training.")
    parser.add_argument("--data_dir", type=str, default="hf://ehsan94/fea-gnn-surrogate",
                        help="Root data directory or hf://owner/repo (default: hf://ehsan94/fea-gnn-surrogate)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=18, help="Hidden dimension")
    parser.add_argument("--mp_steps", type=int, default=3, help="Message passing steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Model save path (default: best_model_<edge_strategy>.pth)")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Tensorboard log directory")
    parser.add_argument("--test_size", type=float, default=0.3, help="Validation split ratio")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="Hugging Face model repo to upload trained model to. "
                             "If not set, no upload is performed. Reads HF_TOKEN from environment.")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = f"best_model_{args.edge_strategy}.pth"

    dataset_path = _dataset_path(args.data_dir, args.mode, args.edge_strategy)
    data_list = load_dataset(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Dataset size: {len(data_list)}")

    train_data, val_data, train_loader, val_loader = create_dataloaders(
        data_list, batch_size=args.batch_size, test_size=args.test_size
    )

    norm_stats = compute_normalization_stats(train_data)
    normalize_data(train_data, norm_stats)
    normalize_data(val_data, norm_stats)

    # Recreate loaders after normalization
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    pos_weight = compute_class_weight(train_data)
    print(f"Positive weight: {pos_weight:.4f}")

    num_features = data_list[0].x.shape[1]
    print(f"Number of features: {num_features}")

    model = SharedMPNN(num_features, args.hidden_dim, 1, args.mp_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=args.epochs,
        save_path=args.save_path,
        log_dir=args.log_dir,
        norm_stats=norm_stats,
        hf_repo=args.hf_repo,
    )

    # Evaluate on test sets if requested
    if args.test_sets:
        from fea_gnn_surrogate.surrogate.inference import load_model
        model, norm_stats = load_model(
            args.save_path, num_features,
            hidden_dim=args.hidden_dim, num_mp_steps=args.mp_steps,
        )
        eval_criterion = torch.nn.BCEWithLogitsLoss()

        print(f"\n{'Test Set':<20}  {'Loss':>8}  {'AUC':>8}")
        print("-" * 40)
        for test_set in args.test_sets:
            test_path = _dataset_path(args.data_dir, test_set, args.edge_strategy)
            test_data = load_dataset(test_path)
            if norm_stats is not None:
                normalize_data(test_data, norm_stats)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
            loss, auc, _, _, _ = validate(model, test_loader, eval_criterion)
            print(f"  {test_set:<18}  {loss:>8.4f}  {auc:>8.4f}")


if __name__ == "__main__":
    main()
