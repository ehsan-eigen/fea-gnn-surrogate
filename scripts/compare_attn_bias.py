"""Compare Graphormer attention-bias variants across multiple seeds.

For each bias in {spd, resistance, resistance_weighted}, trains the model with
early stopping (patience=5 by default) on `--seeds` different seeds and reports
mean ± std of best validation loss, test loss, and test AUC.
"""
import argparse
import os
import random
import statistics
import tempfile

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from fea_gnn_surrogate.surrogate.dataset import (
    load_dataset,
    create_dataloaders,
    compute_class_weight,
    compute_normalization_stats,
    normalize_data,
)
from fea_gnn_surrogate.surrogate.inference import (
    _build_model,
    load_model,
    ATTN_BIAS_CHOICES,
)
from fea_gnn_surrogate.surrogate.train import train, validate


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dataset_path(data_dir, mode, edge_strategy):
    return os.path.join(data_dir, mode, edge_strategy, "dataset", "pyg_line_graphs.pkl")


def run_one(bias_kind, seed, train_pool, test_pool, args, tmp_dir):
    _set_seed(seed)

    # Clone so per-seed normalization doesn't mutate the shared pool.
    pool = [d.clone() for d in train_pool]
    train_data, val_data, _, _ = create_dataloaders(
        pool, batch_size=args.batch_size,
        test_size=args.test_size, random_state=seed,
    )

    norm_stats = compute_normalization_stats(train_data)
    normalize_data(train_data, norm_stats)
    normalize_data(val_data, norm_stats)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    pos_weight = compute_class_weight(train_data)
    num_features = train_pool[0].x.shape[1]

    model = _build_model("graphormer", num_features, args.hidden_dim, 1,
                         args.mp_steps, heads=args.heads,
                         dropout=args.dropout, attn_dropout=args.attn_dropout,
                         max_spd=args.max_spd, attn_bias=bias_kind)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    save_path = os.path.join(tmp_dir, f"{bias_kind}_seed{seed}.pth")
    log_dir = os.path.join(args.log_dir, f"{bias_kind}_seed{seed}_")

    train(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=args.epochs,
        save_path=save_path, log_dir=log_dir,
        norm_stats=norm_stats, hf_repo=None,
        model_type="graphormer", attn_bias=bias_kind,
        early_stop_patience=args.early_stop_patience,
    )

    # Re-load best checkpoint and evaluate on the held-out test pool.
    model, norm_stats_loaded = load_model(
        save_path, num_features,
        hidden_dim=args.hidden_dim, num_mp_steps=args.mp_steps,
        model_type="graphormer", heads=args.heads,
        dropout=args.dropout, attn_dropout=args.attn_dropout,
        max_spd=args.max_spd, attn_bias=bias_kind,
    )
    test_data = [d.clone() for d in test_pool]
    if norm_stats_loaded is not None:
        normalize_data(test_data, norm_stats_loaded)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    eval_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_loss, test_auc, _, _, _ = validate(model, test_loader, eval_criterion)

    # Re-eval val with the best checkpoint, too.
    val_loss, val_auc, _, _, _ = validate(model, val_loader, eval_criterion)

    return {"val_loss": val_loss, "val_auc": val_auc,
            "test_loss": test_loss, "test_auc": test_auc}


def _agg(rows, key):
    vals = [r[key] for r in rows]
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--edge_strategy", type=str, default="no_vn")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--biases", type=str, nargs="+", default=list(ATTN_BIAS_CHOICES),
                   choices=list(ATTN_BIAS_CHOICES))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=18)
    p.add_argument("--mp_steps", type=int, default=3)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--max_spd", type=int, default=20)
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--log_dir", type=str, default="./logs/compare_")
    args = p.parse_args()

    train_path = _dataset_path(args.data_dir, "train", args.edge_strategy)
    test_path = _dataset_path(args.data_dir, "test", args.edge_strategy)
    print(f"Train data: {train_path}")
    print(f"Test data:  {test_path}")
    train_pool = load_dataset(train_path)
    test_pool = load_dataset(test_path)
    print(f"Train size: {len(train_pool)}    Test size: {len(test_pool)}")
    print(f"Biases: {args.biases}    Seeds: {args.seeds}    Patience: {args.early_stop_patience}")
    print()

    results = {b: [] for b in args.biases}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for bias_kind in args.biases:
            for seed in args.seeds:
                print(f"=== bias={bias_kind}  seed={seed} ===")
                r = run_one(bias_kind, seed, train_pool, test_pool, args, tmp_dir)
                print(f"    val_loss={r['val_loss']:.4f}  val_auc={r['val_auc']:.4f}  "
                      f"test_loss={r['test_loss']:.4f}  test_auc={r['test_auc']:.4f}")
                results[bias_kind].append(r)
                print()

    print("=" * 78)
    print(f"Comparison across {len(args.seeds)} seeds (mean ± std)")
    print("=" * 78)
    header = f"{'bias':24s} {'val_loss':>16s} {'test_loss':>16s} {'test_auc':>16s}"
    print(header)
    print("-" * len(header))
    for bias_kind in args.biases:
        rows = results[bias_kind]
        vl_m, vl_s = _agg(rows, "val_loss")
        tl_m, tl_s = _agg(rows, "test_loss")
        ta_m, ta_s = _agg(rows, "test_auc")
        print(f"{bias_kind:24s} "
              f"{vl_m:8.4f} ± {vl_s:5.4f} "
              f"{tl_m:8.4f} ± {tl_s:5.4f} "
              f"{ta_m:8.4f} ± {ta_s:5.4f}")


if __name__ == "__main__":
    main()
