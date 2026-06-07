"""Backfill joint-graph plumbing onto existing base line graphs.

For each base line graph in `data/{mode}/base/line_graphs.pkl`, reads the
corresponding simplified joint graph from `data/{mode}/simplified/graphs/`,
rebuilds the line graph deterministically (`nx.line_graph` + `node_tuple_2_index`)
to recover per-line-node integer joint endpoints, and writes back:

  - node attribute  `endpoints = (u, v)`     per line-graph node
  - graph attribute `num_joints = J`         scalar
  - graph attribute `joint_supports = [...]` sorted joint indices

Verifies attribute consistency between line-graph nodes and Gs edges before
saving so a mismatch in node ordering is caught loudly. After this script,
re-run `scripts/extract_features.py` for each mode to regenerate the PyG
datasets with the new fields.

Idempotent: rerunning just overwrites the same attributes.
"""
import argparse
import os
import pickle

import networkx as nx

from fea_gnn_surrogate.graph.graph_utils import GraphHandler


# Attributes that must match between L.nodes[i] and Gs.edges[endpoints(i)]
# for the line-graph node ordering to be valid.
SANITY_ATTRS = ("dist", "D", "W", "rotation")


def _augment_one(L, simplified_dir):
    name = L.graph["name"]
    Gs_path = os.path.join(simplified_dir, f"{name}.pkl")
    with open(Gs_path, "rb") as f:
        Gs = pickle.load(f)

    # Gs is int-keyed but the IDs are sparse — densify to [0, num_joints).
    gs_to_dense = {node: i for i, node in enumerate(Gs.nodes())}

    L_new = nx.line_graph(Gs)
    # Iteration order of L_new.nodes() matches the original line-graph build
    # because Gs is the same and nx.line_graph is deterministic.
    ordered_endpoints_sparse = list(L_new.nodes())

    if len(ordered_endpoints_sparse) != L.number_of_nodes():
        raise RuntimeError(
            f"graph {name}: line-graph rebuild has {len(ordered_endpoints_sparse)} "
            f"nodes but stored L has {L.number_of_nodes()}"
        )

    for i, (u_sparse, v_sparse) in enumerate(ordered_endpoints_sparse):
        l_attrs = L.nodes[i]
        gs_attrs = Gs.edges[(u_sparse, v_sparse)]
        for attr in SANITY_ATTRS:
            if attr in l_attrs and attr in gs_attrs:
                lv, gv = float(l_attrs[attr]), float(gs_attrs[attr])
                if abs(lv - gv) > 1e-6:
                    raise RuntimeError(
                        f"graph {name} node {i}: attr {attr} mismatch: "
                        f"L={lv} vs Gs.edge[{u_sparse},{v_sparse}]={gv}. "
                        f"Index ordering broke."
                    )
        L.nodes[i]["endpoints"] = (gs_to_dense[u_sparse], gs_to_dense[v_sparse])

    L.graph["num_joints"] = Gs.number_of_nodes()
    L.graph["joint_supports"] = sorted(
        gs_to_dense[j] for j in Gs.nodes() if Gs.nodes[j].get("free") == [0]
    )
    return L


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--modes", nargs="+", default=["train", "test"])
    args = p.parse_args()

    for mode in args.modes:
        base_path = os.path.join(args.data_dir, mode, "base", "line_graphs.pkl")
        simplified_dir = os.path.join(args.data_dir, mode, "simplified", "graphs")
        if not os.path.exists(base_path):
            print(f"[skip {mode}] no base line graphs at {base_path}")
            continue
        if not os.path.isdir(simplified_dir):
            print(f"[skip {mode}] no simplified graphs dir at {simplified_dir}")
            continue

        with open(base_path, "rb") as f:
            line_graphs = pickle.load(f)
        print(f"[{mode}] loaded {len(line_graphs)} line graphs from {base_path}")

        augmented = []
        for k, L in enumerate(line_graphs):
            try:
                augmented.append(_augment_one(L, simplified_dir))
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"missing simplified graph for line graph {k} (name={L.graph.get('name')}). "
                    f"Check that data/{mode}/simplified/graphs/ has all episodes."
                ) from e

        with open(base_path, "wb") as f:
            pickle.dump(augmented, f)
        # Diagnostic on the first augmented graph.
        L0 = augmented[0]
        sample_endpoints = L0.nodes[0].get("endpoints")
        print(f"[{mode}] augmented {len(augmented)} graphs. "
              f"sample[0]: num_joints={L0.graph['num_joints']}, "
              f"|supports|={len(L0.graph['joint_supports'])}, "
              f"node[0].endpoints={sample_endpoints}")


if __name__ == "__main__":
    main()
