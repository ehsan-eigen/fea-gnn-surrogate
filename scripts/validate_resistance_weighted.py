"""Validate that the resistance_weighted underperformance is caused by feeding
z-score-normalized features as 'stiffness' to the EffectiveResistanceBias.

Three variants, identical hyperparameters and seeds:

  weighted_normalized  -- current behavior; reads A_col+A_beam from x AFTER
                          normalize_data has mutated it (suspected bug).
  weighted_raw         -- snapshots A_col+A_beam BEFORE normalization and
                          feeds raw (positive, physical) values via
                          `data.stiffness`.
  constant             -- feeds 1s; equivalent to unweighted resistance.

Prediction (if the hypothesis holds):
  weighted_normalized  >>  weighted_raw  ≈  constant    (in test_loss)
"""
import argparse
import os
import random
import statistics
import tempfile
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree

from fea_gnn_surrogate.surrogate.dataset import (
    load_dataset,
    create_dataloaders,
    compute_class_weight,
    compute_normalization_stats,
    normalize_data,
)
from fea_gnn_surrogate.surrogate.inference import _build_model
from fea_gnn_surrogate.surrogate.model import Graphormer, EffectiveResistanceBias
from fea_gnn_surrogate.surrogate.train import train_one_epoch, validate
from fea_gnn_surrogate.graph.graph_utils import FEATURE_NAMES


A_COL_IDX = FEATURE_NAMES.index("A_col")
A_BEAM_IDX = FEATURE_NAMES.index("A_beam")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _snapshot_stiffness(data_list):
    """Attach `stiffness` (N,) from raw A_col+A_beam. Call BEFORE normalize_data."""
    for d in data_list:
        d.stiffness = (d.x[:, A_COL_IDX].abs() + d.x[:, A_BEAM_IDX].abs() + 1e-6
                       ).detach().clone()


class StiffnessGraphormer(Graphormer):
    """Same as Graphormer but pulls node_w from data.stiffness."""

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        in_deg = degree(edge_index[1], num_nodes=x.size(0)).long().clamp(max=self.max_degree)
        out_deg = degree(edge_index[0], num_nodes=x.size(0)).long().clamp(max=self.max_degree)
        h = self.proj(x) + self.in_degree_emb(in_deg) + self.out_degree_emb(out_deg)
        h_dense, mask = to_dense_batch(h, batch)
        adj = to_dense_adj(edge_index, batch, max_num_nodes=h_dense.size(1))
        node_w, _ = to_dense_batch(data.stiffness, batch)
        bias = self.attn_bias(adj, mask, node_w)
        for layer in self.layers:
            h_dense = layer(h_dense, bias, mask)
        h_dense = self.ln_f(h_dense)
        h_out = h_dense[mask]
        return self.fc_out(h_out)


def build(variant, num_features, args):
    if variant == "weighted_normalized":
        return _build_model("graphormer", num_features, args.hidden_dim, 1,
                            args.mp_steps, heads=args.heads,
                            dropout=args.dropout, attn_dropout=args.attn_dropout,
                            max_spd=args.max_spd, attn_bias="resistance_weighted")
    if variant == "constant":
        return _build_model("graphormer", num_features, args.hidden_dim, 1,
                            args.mp_steps, heads=args.heads,
                            dropout=args.dropout, attn_dropout=args.attn_dropout,
                            max_spd=args.max_spd, attn_bias="resistance")
    if variant == "weighted_raw":
        bias = EffectiveResistanceBias(args.heads)
        return StiffnessGraphormer(num_features, args.hidden_dim, 1,
                                   args.mp_steps, heads=args.heads,
                                   dropout=args.dropout, attn_dropout=args.attn_dropout,
                                   max_spd=args.max_spd, attn_bias=bias)
    raise ValueError(variant)


def run_one(variant, seed, train_pool, test_pool, args):
    _set_seed(seed)
    pool = [d.clone() for d in train_pool]
    _snapshot_stiffness(pool)

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

    model = build(variant, num_features, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    writer = SummaryWriter(args.log_dir + f"{variant}_seed{seed}_" + str(time.time()))
    best_val_loss = float("inf")
    best_state = None
    bad = 0
    for epoch in range(args.epochs):
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion)
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
    _snapshot_stiffness(test_data)
    normalize_data(test_data, norm_stats)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    eval_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_loss, test_auc, _, _, _ = validate(model, test_loader, eval_criterion)
    val_loss, _, _, _, _ = validate(model, val_loader, eval_criterion)
    return {"val_loss": val_loss, "test_loss": test_loss, "test_auc": test_auc}


def _agg(rows, key):
    vals = [r[key] for r in rows]
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--edge_strategy", default="no_vn")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--variants", type=str, nargs="+",
                   default=["weighted_normalized", "weighted_raw", "constant"])
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
    p.add_argument("--log_dir", default="./logs/validate_")
    args = p.parse_args()

    train_path = f"{args.data_dir}/train/{args.edge_strategy}/dataset/pyg_line_graphs.pkl"
    test_path = f"{args.data_dir}/test/{args.edge_strategy}/dataset/pyg_line_graphs.pkl"
    train_pool = load_dataset(train_path)
    test_pool = load_dataset(test_path)
    print(f"Train: {len(train_pool)}  Test: {len(test_pool)}  Seeds: {args.seeds}")

    # Diagnostic: compare raw vs normalized A magnitudes on one sample.
    sample = train_pool[0].clone()
    raw = (sample.x[:, A_COL_IDX].abs() + sample.x[:, A_BEAM_IDX].abs())
    pool_copy = [d.clone() for d in train_pool[:500]]
    ns = compute_normalization_stats(pool_copy)
    normalize_data(pool_copy, ns)
    norm = (pool_copy[0].x[:, A_COL_IDX] + pool_copy[0].x[:, A_BEAM_IDX])
    norm_abs = norm.abs()
    print()
    print("Diagnostic on train_pool[0]:")
    print(f"  raw A_col+A_beam      :  min={raw.min():.4f}  med={raw.median():.4f}  "
          f"max={raw.max():.4f}  std={raw.std():.4f}  (all positive, on physical scale)")
    print(f"  normalized A_col+A_beam:  min={norm.min():.4f}  med={norm.median():.4f}  "
          f"max={norm.max():.4f}  std={norm.std():.4f}  (sign-flipped, mean ≈ 0)")
    print(f"  |normalized| (what bias actually sees): "
          f"min={norm_abs.min():.4f}  med={norm_abs.median():.4f}  max={norm_abs.max():.4f}")
    print()

    results = {v: [] for v in args.variants}
    for variant in args.variants:
        for seed in args.seeds:
            print(f"=== variant={variant}  seed={seed} ===")
            r = run_one(variant, seed, train_pool, test_pool, args)
            print(f"    val_loss={r['val_loss']:.4f}  test_loss={r['test_loss']:.4f}  "
                  f"test_auc={r['test_auc']:.4f}")
            results[variant].append(r)
            print()

    print("=" * 80)
    print(f"Validation across {len(args.seeds)} seeds (mean ± std)")
    print("=" * 80)
    header = f"{'variant':24s} {'val_loss':>16s} {'test_loss':>16s} {'test_auc':>16s}"
    print(header)
    print("-" * len(header))
    for v in args.variants:
        rows = results[v]
        vl_m, vl_s = _agg(rows, "val_loss")
        tl_m, tl_s = _agg(rows, "test_loss")
        ta_m, ta_s = _agg(rows, "test_auc")
        print(f"{v:24s} {vl_m:8.4f} ± {vl_s:5.4f} "
              f"{tl_m:8.4f} ± {tl_s:5.4f} {ta_m:8.4f} ± {ta_s:5.4f}")


if __name__ == "__main__":
    main()
