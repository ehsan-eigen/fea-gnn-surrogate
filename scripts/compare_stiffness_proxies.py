"""Compare stiffness-proxy choices for EffectiveResistanceBias on the line graph.

`A_raw` already lost to `constant` (see compare_attn_bias.py / validate_*.py)
because A_col+A_beam varies <3x within a structure. This script tries proxies
with wider within-graph variation:

  constant   -- node_w = 1; unweighted baseline.
  A_raw      -- node_w = |A_col| + |A_beam|; section area (already known bad).
  AL_raw     -- node_w ≈ A / L; L = span for beams, constant for columns
                (axial stiffness EA/L proxy).
  I_raw      -- node_w = |I_col| + |I_beam|; I ∝ D^3 * W gives ~8x variation
                where A ∝ D * W gives only ~2x.

All proxies are snapshotted from raw, pre-normalization values via
`data.stiffness`, so the prior normalization bug doesn't contaminate this run.
"""
import argparse
import os
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from fea_gnn_surrogate.surrogate.dataset import (
    load_dataset,
    create_dataloaders,
    compute_class_weight,
    compute_normalization_stats,
    normalize_data,
)
from fea_gnn_surrogate.surrogate.inference import _build_model
from fea_gnn_surrogate.surrogate.model import EffectiveResistanceBias
from fea_gnn_surrogate.surrogate.train import train_one_epoch, validate
from fea_gnn_surrogate.graph.graph_utils import FEATURE_NAMES

from validate_resistance_weighted import StiffnessGraphormer  # reuse subclass


IS_COLUMN_IDX = FEATURE_NAMES.index("is_column")
A_COL_IDX = FEATURE_NAMES.index("A_col")
A_BEAM_IDX = FEATURE_NAMES.index("A_beam")
I_COL_IDX = FEATURE_NAMES.index("I_col")
I_BEAM_IDX = FEATURE_NAMES.index("I_beam")
SPAN_IDX = FEATURE_NAMES.index("span")

PROXY_CHOICES = ("constant", "A_raw", "AL_raw", "I_raw")


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _snapshot(data_list, proxy):
    """Attach `stiffness` (N,) from raw features. Must be called BEFORE normalize_data."""
    for d in data_list:
        x = d.x
        is_col = x[:, IS_COLUMN_IDX] > 0.5
        if proxy == "A_raw":
            w = x[:, A_COL_IDX].abs() + x[:, A_BEAM_IDX].abs()
        elif proxy == "AL_raw":
            # Columns: A_col (constant L absorbs into a global scale).
            # Beams:   A_beam / span (≈ EA/L since cos θ ≈ 1).
            a_col = x[:, A_COL_IDX].abs()
            a_beam = x[:, A_BEAM_IDX].abs()
            span = x[:, SPAN_IDX].abs().clamp(min=1e-3)
            w = torch.where(is_col, a_col, a_beam / span)
        elif proxy == "I_raw":
            w = x[:, I_COL_IDX].abs() + x[:, I_BEAM_IDX].abs()
        elif proxy == "constant":
            w = torch.ones(x.size(0))
        else:
            raise ValueError(proxy)
        d.stiffness = (w + 1e-6).detach().clone()


def build_model(num_features, args):
    bias = EffectiveResistanceBias(args.heads)
    return StiffnessGraphormer(num_features, args.hidden_dim, 1,
                               args.mp_steps, heads=args.heads,
                               dropout=args.dropout, attn_dropout=args.attn_dropout,
                               max_spd=args.max_spd, attn_bias=bias)


def run_one(proxy, seed, train_pool, test_pool, args):
    _set_seed(seed)
    pool = [d.clone() for d in train_pool]
    _snapshot(pool, proxy)

    train_data, val_data, _, _ = create_dataloaders(
        pool, batch_size=args.batch_size, test_size=args.test_size, random_state=seed,
    )
    norm_stats = compute_normalization_stats(train_data)
    normalize_data(train_data, norm_stats)
    normalize_data(val_data, norm_stats)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    pos_weight = compute_class_weight(train_data)
    num_features = train_pool[0].x.shape[1]

    model = build_model(num_features, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    writer = SummaryWriter(args.log_dir + f"{proxy}_seed{seed}_" + str(time.time()))
    best_val_loss = float("inf"); best_state = None; bad = 0
    for epoch in range(args.epochs):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc, _, _, _ = validate(model, val_loader, criterion)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("AUC/Validation", val_auc, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.early_stop_patience:
                break
    writer.close()
    model.load_state_dict(best_state)

    test_data = [d.clone() for d in test_pool]
    _snapshot(test_data, proxy)
    normalize_data(test_data, norm_stats)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    eval_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_loss, test_auc, _, _, _ = validate(model, test_loader, eval_criterion)
    val_loss, _, _, _, _ = validate(model, val_loader, eval_criterion)
    return {"val_loss": val_loss, "test_loss": test_loss, "test_auc": test_auc}


def _agg(rows, key):
    vals = [r[key] for r in rows]
    return (vals[0], 0.0) if len(vals) == 1 else (statistics.mean(vals), statistics.stdev(vals))


def _proxy_diagnostic(train_pool):
    """Print within-graph variation for each proxy on sample[0]."""
    sample = train_pool[0].clone()
    proxies = ("A_raw", "AL_raw", "I_raw")
    print("Per-proxy within-graph variation on train_pool[0]:")
    print(f"  {'proxy':10s}  {'min':>10s}  {'med':>10s}  {'max':>10s}  {'max/min':>10s}")
    for p in proxies:
        s = [sample.clone()]
        _snapshot(s, p)
        w = s[0].stiffness
        ratio = (w.max() / w.min().clamp(min=1e-12)).item()
        print(f"  {p:10s}  {w.min():10.4f}  {w.median():10.4f}  {w.max():10.4f}  {ratio:10.2f}x")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--edge_strategy", default="no_vn")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--proxies", type=str, nargs="+", default=list(PROXY_CHOICES),
                   choices=list(PROXY_CHOICES))
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
    p.add_argument("--log_dir", default="./logs/proxy_")
    args = p.parse_args()

    train_pool = load_dataset(f"{args.data_dir}/train/{args.edge_strategy}/dataset/pyg_line_graphs.pkl")
    test_pool = load_dataset(f"{args.data_dir}/test/{args.edge_strategy}/dataset/pyg_line_graphs.pkl")
    print(f"Train: {len(train_pool)}  Test: {len(test_pool)}  Seeds: {args.seeds}")
    print()
    _proxy_diagnostic(train_pool)

    results = {p: [] for p in args.proxies}
    for proxy in args.proxies:
        for seed in args.seeds:
            print(f"=== proxy={proxy}  seed={seed} ===")
            r = run_one(proxy, seed, train_pool, test_pool, args)
            print(f"    val_loss={r['val_loss']:.4f}  test_loss={r['test_loss']:.4f}  "
                  f"test_auc={r['test_auc']:.4f}")
            results[proxy].append(r)
            print()

    print("=" * 80)
    print(f"Stiffness-proxy comparison across {len(args.seeds)} seeds (mean ± std)")
    print("=" * 80)
    header = f"{'proxy':12s} {'val_loss':>16s} {'test_loss':>16s} {'test_auc':>16s}"
    print(header); print("-" * len(header))
    for proxy in args.proxies:
        rows = results[proxy]
        vl_m, vl_s = _agg(rows, "val_loss")
        tl_m, tl_s = _agg(rows, "test_loss")
        ta_m, ta_s = _agg(rows, "test_auc")
        print(f"{proxy:12s} {vl_m:8.4f} ± {vl_s:5.4f} "
              f"{tl_m:8.4f} ± {tl_s:5.4f} {ta_m:8.4f} ± {ta_s:5.4f}")


if __name__ == "__main__":
    main()
