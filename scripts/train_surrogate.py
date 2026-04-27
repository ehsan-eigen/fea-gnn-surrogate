"""Train the GNN surrogate model."""
import argparse
import torch

from structural_analysis.surrogate.model import SharedMPNN
from structural_analysis.surrogate.dataset import (
    load_dataset,
    create_dataloaders,
    compute_class_weight,
    compute_normalization_stats,
    normalize_data,
)
from structural_analysis.surrogate.train import train


def main():
    parser = argparse.ArgumentParser(description="Train GNN surrogate model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to pyg_line_graphs.pkl")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=18, help="Hidden dimension")
    parser.add_argument("--mp_steps", type=int, default=3, help="Message passing steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Model save path")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Tensorboard log directory")
    parser.add_argument("--test_size", type=float, default=0.3, help="Validation split ratio")
    args = parser.parse_args()

    data_list = load_dataset(args.dataset_path)
    print(f"Dataset size: {len(data_list)}")

    train_data, val_data, train_loader, val_loader = create_dataloaders(
        data_list, batch_size=args.batch_size, test_size=args.test_size
    )

    norm_stats = compute_normalization_stats(train_data)
    normalize_data(train_data, norm_stats)
    normalize_data(val_data, norm_stats)

    # Recreate loaders after normalization
    from torch_geometric.loader import DataLoader
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
    )


if __name__ == "__main__":
    main()
