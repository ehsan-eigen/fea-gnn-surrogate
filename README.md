# 2D Frame GNN Surrogate

![Cover](cover_photo.png)

A pip-installable Python package for generating, analysing, and learning from 2D multi-story reinforced concrete frame structures.

---

## Problem Statement

Assume we want to assess the stability of a frame structure under the following conditions:

**Stability Criteria:**
- Lateral drift: $H/500$
- Vertical deflection: $L/2000$

**Modulus of Elasticity (MOE):**
- 32800 MPa

**Load:**
- Lateral Load: Point load on each level representing wind load from left to right, increasing with height.
- Vertical Load: SDL, uniform (300 KN/m)

In the above image, beams and columns that do not meet the stability criteria are colored red (invalid elements). The blue elements are considered valid. A structure is valid if and only if all its elements are valid (i.e., the deflection of beams and drift of columns are within the specified thresholds).

## Key aspects of this problem:

1. **Validation Threshold:** We predict whether deformation stays within code-specified limits (not the exact value), since structural engineers care about pass/fail, not precise magnitudes.

2. **Multiple Loads:** The structure is subjected to both vertical (SDL) and lateral (wind) loads simultaneously.

3. **Scoped Generalization:** A fully general surrogate for arbitrary frame geometries is intractable — it would require billions of samples. Instead, we focus on a specific family: grid-layout RC frames with two basement levels and up to 11 above-ground residential levels, where the basement column layout may differ from the upper levels (requiring transfer beams at the first floor).

---

## GNN Architecture — The Core Innovation

> **The GNN model and its graph data representation are the central contribution of this project.**

Predicting structural validity is not a standard supervised learning problem. Because frame structures are connected systems, the deflection of any element depends on the stiffness and loading of the *entire* structure — not just its own cross-section. This rules out per-element classifiers. Three key design decisions make the surrogate tractable:

### 1. Line Graph Transformation

Frame structures are naturally edge-centric: beams and columns are edges, joints are nodes. Edge-level GNN prediction is harder than node-level prediction, so each structure is converted to its **line graph** — original edges become nodes, and two line-graph nodes are connected if the corresponding elements share a joint. This turns element-validity prediction into standard node classification.

Before the transformation, all joint information (support conditions, relative position) is embedded into adjacent edge features so no structural information is lost.

### 2. Physically Motivated Feature Engineering

Raw cross-section dimensions are replaced by features grounded in structural mechanics:

| Feature | Physical meaning |
|---|---|
| **DW** | Proportional to cross-sectional area (axial stiffness) |
| **D³W** | Proportional to second moment of area (bending stiffness) |
| **Transfer beam flag** | Identifies elements carrying redistributed column loads |
| **Cantilever flag** | Flags elements with a free end (high deflection risk) |
| **Level & position ratios** | Encode location in the load path |
| **Laplacian PE (k=8)** | Spectral coordinates from the normalized graph Laplacian — encode each element's position within the load-path topology |

Each node in the line graph has 21 features (13 structural + 8 Laplacian positional encoding). Feature names and normalization targets are defined in `graph_utils.py` as `FEATURE_NAMES`, `NORM_COLUMN_FEATURES`, and `NORM_BEAM_FEATURES`.

### 3. Shared-Weight Message Passing (SharedMPNN)

A single `GraphConv` layer is applied repeatedly for `mp_steps` iterations with **tied weights**. This forces the model to learn one general message-passing rule rather than separate per-layer transformations — a strong inductive bias that matches the structural regularity of frame buildings.

### 4. GPS Graph Transformer (GPSModel)

An alternative architecture based on the **General, Powerful, Scalable (GPS)** Graph Transformer (Rampasek et al., NeurIPS 2022). Each GPS layer runs a local `GINConv` and a global multi-head self-attention in parallel, then combines their outputs:

- **Local branch:** `GINConv` aggregates information from graph neighbors (same local inductive bias as SharedMPNN)
- **Global branch:** multi-head self-attention attends over all nodes simultaneously — every element can directly attend to every other element, regardless of topological distance

This eliminates the need for artificial edges (e.g. virtual node) to propagate long-range structural dependencies. The model is trained on the `no_vn` edge strategy (plain line graph, no added edges).

Select the GPS architecture with `--model gps` when training.

---

### Read more

**[→ GNN Model Intuitions](GNN_model_intuitions.md)** — a detailed walkthrough of the data representation, line graph construction, feature derivations, message passing mechanics, and directions for improvement. This is the primary technical reference for the model.

---

## Installation

Requires Python >= 3.9.

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch
pip install torch_geometric
pip install -e .
```

`huggingface_hub` is included as a dependency and installed automatically. To upload datasets or models to Hugging Face, set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=your_token_here
```

---

## Quick Start

### 1. Generate data

```bash
# Training data (1000 structures) — saved locally and uploaded to Hugging Face
python scripts/generate_dataset.py --mode train --num_samples 1000

# Test data
python scripts/generate_dataset.py --mode test_1 --num_samples 500
python scripts/generate_dataset.py --mode test_2 --num_samples 500
```

Each run:
- Generates structures, runs FEA, and saves raw NetworkX line graphs to `data/{mode}/base/` (the **base dataset** — topology + attributes, no feature extraction)
- Extracts PyG features and saves two edge-strategy variants (`with_vn`, `no_vn`) from the same structures

By default the generated `.pkl` files are uploaded to the `ehsan94/fea-gnn-surrogate` Hugging Face dataset repository (requires `HF_TOKEN`). Pass `--hf_repo ""` to skip the upload.

**Changing features without re-running FEA:** If you modify feature extraction (e.g. add or remove a feature), you can re-extract from the saved base data without regenerating structures or re-running FEA:

```bash
python scripts/extract_features.py --mode train
python scripts/extract_features.py --mode test_1
```

### 2. Train

```bash
# SharedMPNN (default)
python scripts/train_surrogate.py --mode train --edge_strategy with_vn --epochs 100

# GPS Graph Transformer — no artificial edges needed
python scripts/train_surrogate.py --mode train --model gps --edge_strategy no_vn --epochs 100
```

Training data is loaded from Hugging Face by default (`hf://ehsan94/fea-gnn-surrogate`). To use a local directory instead, pass `--data_dir data`.

The best model is saved to `best_model_<edge_strategy>.pth` and uploaded to the Hugging Face model repository specified by `--hf_repo` (requires `HF_TOKEN`). To also report test metrics after training:

```bash
python scripts/train_surrogate.py --mode train --model gps --edge_strategy no_vn --epochs 100 \
    --test_sets test_1 test_2 test_3
```

### 3. Evaluate on test sets

```bash
python scripts/evaluate_model.py --model_path best_model_with_vn.pth \
    --test_sets test_1 test_2
```

Test data is loaded from Hugging Face by default. Both `--model_path` and `--data_dir` accept `hf://owner/repo` paths:

```bash
python scripts/evaluate_model.py \
    --model_path hf://ehsan94/fea-gnn-surrogate/best_model_with_vn.pth \
    --test_sets test_1 test_2
```

Output:

```
Model:  best_model_with_vn.pth
Test Set              Loss       AUC
----------------------------------------
  test_1              0.2265    0.9724
  test_2              0.0717    0.9939
```

### 4. Inference — rank structures

```bash
python scripts/run_inference.py --test_name test_1
```

This loads the model matching the edge strategy (default: `best_model_with_vn.pth`), ranks test structures by predicted validity, runs FEA on the top candidates, and saves deflection plots.

---

## Configuration

`config.json` defines the building geometry for each dataset. It contains these sections:

- `train` — the training layout (all column positions available, storey count randomised during generation)
- `test_1` through `test_5` — five different test layouts with specific column positions

Each section has these fields:

| Field | Meaning |
|---|---|
| `num_cols` | Grid width (number of column positions) |
| `num_rows` | Number of storeys (randomised +-3 during training) |
| `transfer_row` | Storey index of the transfer slab |
| `horizontal_scale` | Bay width in metres per grid unit |
| `vertical_scale` | Storey height in metres |
| `possible_columns_up` | Candidate column positions above the transfer slab |
| `possible_columns_down` | Candidate column positions below the transfer slab |
| `distance_lower_bound` | Minimum allowable column spacing (metres) |
| `distance_upper_bound` | Maximum allowable span (metres) |

---

## CLI Reference

### generate_dataset.py

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | Config section: `train`, `test_1`–`test_5` |
| `--num_samples` | `1000` | Number of structures to generate |
| `--config` | `config.json` | Path to the configuration file |
| `--output_dir` | `data` | Root output directory |
| `--output_name` | `pyg_line_graphs.pkl` | Filename for the saved PyG dataset |
| `--visualize` | off | Save a deflection plot for each structure |
| `--skip_fea` | off | Skip FEA (no labels — cannot train) |
| `--no_save_graphs` | off | Skip saving raw/simplified NetworkX graphs |
| `--hf_repo` | `ehsan94/fea-gnn-surrogate` | Hugging Face dataset repo to upload to (reads `HF_TOKEN`); pass `""` to skip |

Always saves base NX line graphs to `data/{mode}/base/` alongside the PyG variants.

### extract_features.py

Re-extract PyG features from saved base NX line graphs. Use this when changing features — avoids re-running FEA.

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | Config section (must match a previous `generate_dataset.py` run) |
| `--data_dir` | `data` | Root data directory |
| `--output_name` | `pyg_line_graphs.pkl` | Filename for the saved PyG dataset |
| `--skip_fea` | off | Set if the base graphs have no labels (generated with `--skip_fea`) |

### train_surrogate.py

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | Config section for training data |
| `--model` | `mpnn` | Model architecture: `mpnn` (SharedMPNN) or `gps` (GPS Transformer) |
| `--edge_strategy` | `with_vn` | `with_vn` or `no_vn` |
| `--test_sets` | none | Test sets to evaluate after training (e.g. `test_1 test_2`) |
| `--data_dir` | `hf://ehsan94/fea-gnn-surrogate` | Root data directory or `hf://owner/repo` |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--hidden_dim` | `18` | Hidden dimension |
| `--mp_steps` | `3` | Shared message-passing steps (MPNN) or GPS layers (GPS) |
| `--heads` | `3` | Attention heads — GPS only; `hidden_dim` must be divisible by `heads` |
| `--dropout` | `0.2` | Dropout rate — GPS only |
| `--attn_dropout` | `0.2` | Attention dropout rate — GPS only |
| `--lr` | `0.01` | Learning rate |
| `--test_size` | `0.3` | Validation split ratio |
| `--save_path` | `best_model_<edge_strategy>.pth` | Model checkpoint path |
| `--log_dir` | `./logs/` | TensorBoard log directory |
| `--hf_repo` | `ehsan94/fea-gnn-surrogate` | Hugging Face model repo to upload trained model to (reads `HF_TOKEN`) |

### evaluate_model.py

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `best_model_<edge_strategy>.pth` | Path to trained model checkpoint; accepts `hf://owner/repo/file.pth` |
| `--test_sets` | (required) | Test set names (e.g. `test_1 test_2`) |
| `--edge_strategy` | `with_vn` | Edge strategy variant |
| `--data_dir` | `hf://ehsan94/fea-gnn-surrogate` | Root data directory or `hf://owner/repo` |
| `--hidden_dim` | `18` | Must match training |
| `--mp_steps` | `3` | Must match training |
| `--heads` | `3` | Must match training (GPS only) |
| `--dropout` | `0.2` | Must match training (GPS only) |
| `--attn_dropout` | `0.2` | Must match training (GPS only) |
| `--batch_size` | `32` | Batch size for evaluation |

### run_inference.py

| Argument | Default | Description |
|---|---|---|
| `--test_name` | (required) | Test case name (e.g. `test_1`) |
| `--edge_strategy` | `with_vn` | Edge strategy variant |
| `--model_path` | `best_model_<edge_strategy>.pth` | Model checkpoint path |
| `--data_dir` | `data` | Root data directory |
| `--top_stability` | `15` | Top-K by predicted validity |
| `--top_weight` | `15` | Top-K lightest from those |
| `--output_dir` | `top_structs` | Deflection plot output directory |
| `--hidden_dim` | `18` | Must match training |
| `--mp_steps` | `3` | Must match training |

---

## Edge Strategy Comparison

Each generation run produces two graph representations of the **same** structures:

| Variant | Description |
|---|---|
| **`with_vn`** | Global shared virtual node connected to all elements — every element is exactly 2 hops from every other |
| **`no_vn`** | No artificial edges; pure local message passing (baseline) |

### Experimental results (with Laplacian PE, k=8)

#### SharedMPNN

| Variant | Test 1 AUC | Test 1 Loss | Test 2 AUC | Test 2 Loss | Test 3 AUC | Test 3 Loss |
|---|---|---|---|---|---|---|
| No edges (baseline) | 0.9830 | 0.1698 | 0.9980 | 0.0436 | 0.9985 | 0.0560 |
| Virtual node | 0.9836 | 0.2031 | 0.9979 | 0.0408 | 0.9974 | 0.0613 |

With Laplacian positional encoding, both SharedMPNN variants perform similarly. The spectral coordinates capture each element's position within the load-path topology, which was previously the main benefit of virtual nodes.

#### GPS Graph Transformer

| Variant | Test 1 AUC | Test 1 Loss | Test 2 AUC | Test 2 Loss | Test 3 AUC | Test 3 Loss |
|---|---|---|---|---|---|---|
| GPS (no edges) | **0.9813** | **0.1754** | **0.9985** | **0.0262** | **0.9992** | **0.0214** |

GPS is trained on the plain line graph (`no_vn`) — no virtual node needed. Global self-attention replaces all hand-crafted long-range connections. On test_2 and test_3 it substantially outperforms both SharedMPNN variants, matching or exceeding the best MPNN result on test_1 as well.

---

## Project Structure

```
fea-gnn-surrogate/
├── pyproject.toml                        # pip package definition
├── config.json                           # geometry configs for train + test sets
│
├── src/fea_gnn_surrogate/               # main package
│   ├── config.py                         # load_config(): parse config.json
│   ├── generate.py                       # generate_samples(): full generation pipeline
│   ├── visualization.py                  # plot_structure(), plot_deflection()
│   ├── fea/
│   │   ├── geometry.py                   # member_rotation, calc_rotation_length
│   │   ├── stiffness.py                  # build_K, calculate_Kg, calc_DOF
│   │   ├── nodal_reactions.py            # calc_displacement, assemble_UG
│   │   ├── member_reactions.py           # transform_U_to_local, deflection diagrams
│   │   └── environment.py               # StructEnvironment: analyse(), set_attributes()
│   ├── graph/
│   │   └── graph_utils.py               # GraphHandler: generate, simplify, PyG serialisation
│   └── surrogate/
│       ├── model.py                      # SharedMPNN, GPSModel, FC model definitions
│       ├── dataset.py                    # load_dataset, create_dataloaders, normalisation
│       ├── train.py                      # training loop with TensorBoard logging
│       └── inference.py                  # load_model, predict, rank_structures
│
├── scripts/
│   ├── generate_dataset.py               # CLI: generate samples + save base NX graphs + PyG graphs
│   ├── extract_features.py               # CLI: re-extract PyG features from base NX graphs (no FEA)
│   ├── train_surrogate.py                # CLI: train the GNN
│   ├── evaluate_model.py                 # CLI: report test set metrics
│   └── run_inference.py                  # CLI: rank test structures by GNN predictions
│
└── tests/
    └── __init__.py
```

---

## Stability Criteria

An element is labelled **valid** (`y=1`) if it satisfies:

- **Beam vertical deflection:** normalised deflection < L / 2000, where L is the span length
- **Column lateral drift:** inter-storey drift < H / 500, where H is the storey height

A structure is considered valid only if **all** of its elements are valid.

---

## Programmatic API

```python
from fea_gnn_surrogate.generate import generate_samples
from fea_gnn_surrogate.graph.graph_utils import GraphHandler
from fea_gnn_surrogate.surrogate.inference import load_model, predict, rank_structures
from fea_gnn_surrogate.surrogate.dataset import load_dataset, normalize_data

# Generate 100 training samples — returns a list of NetworkX line graphs
line_graphs = generate_samples(
    config_path="config.json", mode="train", num_episodes=100
)

# Save base NX line graphs (topology + attributes) — re-use when features change
GraphHandler.save_base_line_graphs(line_graphs, "data/train/base", "line_graphs.pkl")

# Extract PyG features and save both edge-strategy variants
GraphHandler.save_pyg_line_graphs(
    line_graphs, "data/train/with_vn/dataset", "pyg_line_graphs.pkl",
    use_virtual_node=True,
)
GraphHandler.save_pyg_line_graphs(
    line_graphs, "data/train/no_vn/dataset", "pyg_line_graphs.pkl",
    use_virtual_node=False,
)

# Load a test dataset and run inference (local path or hf:// URI both work)
test_data = load_dataset("hf://ehsan94/fea-gnn-surrogate/test_1/with_vn/dataset/pyg_line_graphs.pkl")

model, norm_stats = load_model("hf://ehsan94/fea-gnn-surrogate/best_model_with_vn.pth", num_features=21)
if norm_stats:
    normalize_data(test_data, norm_stats)

results = predict(model, test_data)
df = rank_structures(results)
print(df)
```

---

## Related Resources

The following tutorials are published on [Engineering Skills](https://www.engineeringskills.com). Readers who are new to machine learning are encouraged to go through them in the order listed below, as each one builds on the concepts introduced in the previous.

1. [Machine Learning in Civil Engineering: Sensitivity Analysis](https://www.engineeringskills.com/posts/members/machine-learning-in-civil-engineering-sensitivity-analysis) — A gentle introduction to optimisation methods in civil engineering, with a focus on the adjoint method for sensitivity analysis. A good starting point if you are new to the topic.

2. [Machine Learning in Civil Engineering: Surrogate Models](https://www.engineeringskills.com/posts/members/machine-learning-in-civil-engineering-surrogate-models) — Introduces the concept of surrogate modelling: replacing expensive simulations with lightweight learned functions. Covers supervised learning fundamentals, regression, and classification through a truss deflection example.

3. [Advanced Surrogate Models with Graph Neural Networks](https://www.engineeringskills.com/posts/members/advanced-surrogate-models-with-graph-neural-networks) — A comprehensive companion tutorial to this project. Covers the full pipeline in depth: graph representation of frame structures, line graph transformation, feature engineering, GNN message passing, model training, and evaluation with detailed analysis and visualisations.
