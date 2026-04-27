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
Key aspects of this problem:

1. **Validation Threshold:** Instead of estimating the deformation itself, we assess whether the deformation satisfies a valid threshold (maximum normal deflection for beams and maximum drift for columns). Structural engineers generally do not focus on the exact deformation as long as it is within the standard range specified in civil codes.

2. **Multiple Loads:** The structure is subjected to multiple vertical and lateral load forces.

3. **Generalized Function:** Our objective is to create a function that can estimate the deflection for any frame structure with arbitrary geometry (arbitrary stiffness matrix) and under arbitrary load vectors. However, the diversity of possible structures that can be modeled with frame elements is vast, and the relationship between design variables and deformation can be highly complex in the general case. Capturing this diversity and designing a surrogate model that accurately approximates the deflection of any structure would require generating billions of samples for various frame designs and loads to extract meaningful patterns. This task is beyond the scope of this project.

    Therefore, we focus on a family of frame structures with specific constraints on their geometrical designs and force loads. Our goal is to analyze the stability of grid layout frame structures where the first two levels have the same slab plan and column layout, which differs from the slab plan of the upper levels. In other words, the grid layout is consistent within the two lower levels and within all the upper levels, but not necessarily between them. These designs resemble a simplified version of a concrete frame building with two basement levels (for parking) and an arbitrary number of above-ground residential levels. Therefore, the first-floor beams are typically transfer beams. We have limited our samples to a maximum of 11 above-ground levels. The pattern of lateral and vertical loads is depicted in the image above: the lateral load increases with height and is only applied to the above-ground levels, while the vertical load is uniformly distributed (300 KN/m) on each level.

For a deeper discussion of the GNN model architecture, message passing, feature engineering, and ideas for improvement, see [GNN Model Intuitions](GNN_model_intuitions.md).

---

## Pipeline Overview

The pipeline has three stages:

1. **Generate** — randomly sample frame structures from a geometry config and run finite element analysis (FEA) to label each beam/column as valid or invalid based on vertical deflection and lateral drift
2. **Train** — train a Graph Neural Network (GNN) on the labelled graphs to predict element validity without running FEA
3. **Infer** — use the trained GNN to rank unseen structures by predicted stability and visualize the best candidates

---

## Project structure

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
│       ├── model.py                      # SharedMPNN, FC model definitions
│       ├── dataset.py                    # load_dataset, create_dataloaders, normalisation
│       ├── train.py                      # training loop with TensorBoard logging
│       └── inference.py                  # load_model, predict, rank_structures
│
├── scripts/
│   ├── generate_dataset.py               # CLI: generate samples + save PyG graphs
│   ├── train_surrogate.py                # CLI: train the GNN
│   └── run_inference.py                  # CLI: rank test structures by GNN predictions
│
└── tests/
    └── __init__.py
```

---

## Installation

Requires Python >= 3.9.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install PyTorch (CPU example; see https://pytorch.org for CUDA versions)
pip install torch

# Install PyTorch Geometric
pip install torch_geometric

# Install this package and remaining dependencies in editable mode
pip install -e .
```

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

## Quick start — full pipeline

Below is the complete sequence of commands to go from raw data to ranked structures. Each stage is explained in detail in the sections that follow.

```bash
# Stage 1a — Generate 1000 labelled training structures (--save_graphs needed for the tutorial notebook)
python scripts/generate_dataset.py --mode train --num_samples 1000 --save_graphs

# Stage 1b — Generate 500 test structures (with --save_graphs for visualization in Stage 3)
python scripts/generate_dataset.py --mode test_1 --num_samples 500 --save_graphs

# Stage 2 — Train the GNN surrogate on the training data
python scripts/train_surrogate.py \
    --dataset_path data/train/dataset/pyg_line_graphs.pkl \
    --epochs 100

# Stage 3 — Run inference on the test data: rank structures and produce deflection plots
python scripts/run_inference.py --test_name test_1
```

---

## Stage 1 — Generate datasets

You need to generate **two** datasets: one for training and one for testing. Both run FEA to label every beam and column as valid or invalid.

### 1a. Generate training data

```bash
python scripts/generate_dataset.py --mode train --num_samples 1000 --save_graphs
```

This generates 1000 random frame structures using the `train` section of `config.json`, runs FEA on each, labels every element, and saves the result as a PyG dataset to:

```
data/train/dataset/pyg_line_graphs.pkl
```

### 1b. Generate test data

```bash
python scripts/generate_dataset.py --mode test_1 --num_samples 500 --save_graphs
```

This does the same using the `test_1` geometry. The `--save_graphs` flag saves the raw and simplified NetworkX graphs alongside the PyG dataset:

```
data/test_1/dataset/pyg_line_graphs.pkl      # PyG dataset (used by Stage 2 and 3)
data/test_1/raw/graphs/0.pkl, 1.pkl, ...     # raw NetworkX graphs
data/test_1/simplified/graphs/0.pkl, ...     # simplified NetworkX graphs
```

The raw and simplified graphs are needed in Stage 3 to re-run FEA on the top-ranked structures and produce deflection plots.

You can generate data for any test section (`test_2`, `test_3`, etc.) by changing `--mode`.

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | Which section of `config.json` to use: `train`, `test_1`, `test_2`, `test_3`, `test_4`, `test_5` |
| `--num_samples` | `1000` | Number of structures to generate |
| `--config` | `config.json` | Path to the configuration file |
| `--output_dir` | `data` | Root output directory |
| `--output_name` | `pyg_line_graphs.pkl` | Filename for the saved PyG dataset |
| `--visualize` | off | Save a deflection plot (PNG) for each generated structure |
| `--skip_fea` | off | Skip FEA. Structures will have no validity labels and cannot be used for training |
| `--save_graphs` | off | Also save raw and simplified NetworkX graphs. Required for deflection plots in Stage 3 |

---

## Stage 2 — Train the GNN surrogate

Train the model on the dataset generated in Stage 1a:

```bash
python scripts/train_surrogate.py \
    --dataset_path data/train/dataset/pyg_line_graphs.pkl \
    --epochs 100 \
    --save_path best_model.pth
```

The checkpoint contains both the model weights and the normalisation statistics from the training split, so inference does not need a separate stats file.

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_path` | (required) | Path to the training `.pkl` dataset |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--hidden_dim` | `18` | Hidden dimension of GNN layers |
| `--mp_steps` | `3` | Number of shared-weight message-passing steps |
| `--lr` | `0.01` | Adam learning rate |
| `--test_size` | `0.3` | Fraction of data held out for validation |
| `--save_path` | `best_model.pth` | Where to save the best model checkpoint |
| `--log_dir` | `./logs/` | TensorBoard log directory |

Monitor training:

```bash
tensorboard --logdir logs/
```

### Model architecture

`SharedMPNN` is a lightweight message-passing network:

1. **Linear projection** — maps 13 input features to `hidden_dim`
2. **Shared GraphConv** — applied `mp_steps` times with the same weights (weight-tied message passing)
3. **Linear readout** — maps `hidden_dim` to 1 logit per node (element)

Training uses `BCEWithLogitsLoss` with a class-rebalancing `pos_weight` to handle the imbalance between valid (majority) and invalid elements.

---

## Stage 3 — Inference on test structures

Use the trained model to rank test structures by predicted validity, then re-run FEA on the best candidates to produce deflection plots:

```bash
python scripts/run_inference.py \
    --test_name test_1 \
    --model_path best_model.pth
```

**Prerequisite:** You must have generated the test dataset in Stage 1b with `--save_graphs`.

### What this does

1. Loads the test PyG dataset and normalises features using the stats stored in the checkpoint
2. Runs the GNN to compute a validity score for each structure (minimum predicted probability across all elements)
3. Ranks structures by validity score, then filters by weight to find the lightest valid designs
4. Re-runs full FEA on the top structures and saves deflection plots to `--output_dir`

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--test_name` | (required) | Config section for the test case (`test_1`, `test_2`, etc.) |
| `--model_path` | `best_model.pth` | Path to trained model checkpoint |
| `--data_dir` | `data` | Root data directory, must match `--output_dir` used in Stage 1 |
| `--dataset_path` | auto | Override the PyG dataset path (by default derived from `--data_dir` and `--test_name`) |
| `--config` | `config.json` | Path to the configuration file |
| `--top_stability` | `15` | Keep top-K structures by predicted validity score |
| `--top_weight` | `15` | From those, keep the top-K lightest structures |
| `--output_dir` | `top_structs` | Directory for deflection plot output |
| `--hidden_dim` | `18` | Must match the value used during training |
| `--mp_steps` | `3` | Must match the value used during training |

---

## Programmatic API

```python
from fea_gnn_surrogate.generate import generate_samples
from fea_gnn_surrogate.graph.graph_utils import GraphHandler
from fea_gnn_surrogate.surrogate.inference import load_model, predict, rank_structures
from fea_gnn_surrogate.surrogate.dataset import normalize_data
import pickle

# Generate 100 training samples
line_graphs = generate_samples(config_path="config.json", mode="train", num_episodes=100)

# Save as PyG dataset
GraphHandler.save_pyg_line_graphs(
    line_graphs, "data/train/dataset", "pyg_line_graphs.pkl"
)

# Load a test dataset and run inference
with open("data/test_1/dataset/pyg_line_graphs.pkl", "rb") as f:
    test_data = pickle.load(f)

model, norm_stats = load_model("best_model.pth", num_features=13)
if norm_stats:
    normalize_data(test_data, norm_stats)

results = predict(model, test_data)
df = rank_structures(results)
print(df)
```

---

## Stability criteria

An element is labelled **valid** (`y=1`) if it satisfies:

- **Beam vertical deflection:** normalised deflection < L / 2000, where L is the span length
- **Column lateral drift:** inter-storey drift < H / 500, where H is the storey height

A structure is considered valid only if **all** of its elements are valid.

---

## Pre-trained model

A pre-trained checkpoint `best_model.pth` is included in the repository, trained on 1000 samples.

---

## Related resources

The following tutorials are published on [Engineering Skills](https://www.engineeringskills.com). Readers who are new to machine learning are encouraged to go through them in the order listed below, as each one builds on the concepts introduced in the previous.

1. [Machine Learning in Civil Engineering: Sensitivity Analysis](https://www.engineeringskills.com/posts/members/machine-learning-in-civil-engineering-sensitivity-analysis) — A gentle introduction to optimisation methods in civil engineering, with a focus on the adjoint method for sensitivity analysis. A good starting point if you are new to the topic.

2. [Machine Learning in Civil Engineering: Surrogate Models](https://www.engineeringskills.com/posts/members/machine-learning-in-civil-engineering-surrogate-models) — Introduces the concept of surrogate modelling: replacing expensive simulations with lightweight learned functions. Covers supervised learning fundamentals, regression, and classification through a truss deflection example.

3. [Advanced Surrogate Models with Graph Neural Networks](https://www.engineeringskills.com/posts/members/advanced-surrogate-models-with-graph-neural-networks) — A comprehensive companion tutorial to this project. Covers the full pipeline in depth: graph representation of frame structures, line graph transformation, feature engineering, GNN message passing, model training, and evaluation with detailed analysis and visualisations.
