import os
import pickle
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from fea_gnn_surrogate.graph.graph_utils import (
    FEATURE_NAMES,
    NORM_COLUMN_FEATURES,
    NORM_BEAM_FEATURES,
    NORM_LOAD_FEATURES,
    feature_indices,
)


def _resolve_dataset_path(path):
    """If path uses hf://owner/repo/path/file.pkl, check locally first, then download."""
    if not path.startswith("hf://"):
        return path
    # hf://owner/repo-name/a/b/c.pkl  →  repo_id="owner/repo-name", filename="a/b/c.pkl"
    parts = path[len("hf://"):].split("/")
    if len(parts) < 3:
        raise ValueError("hf:// dataset paths must be hf://owner/repo-name/path/to/file.pkl")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    # Check locally before attempting a network download
    for local_root in ["data", "."]:
        local_path = os.path.join(local_root, filename)
        if os.path.exists(local_path):
            print(f"Using local file: {local_path}")
            return local_path
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    return local_path


def load_dataset(path):
    local_path = _resolve_dataset_path(path)
    with open(local_path, "rb") as f:
        pyg_data_list = pickle.load(f)
    return pyg_data_list


def create_dataloaders(data_list, batch_size=32, test_size=0.3, random_state=20):
    train_data, val_data = train_test_split(
        data_list, test_size=test_size, random_state=random_state, shuffle=True
    )
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_data, val_data, train_dataloader, val_dataloader


def compute_class_weight(train_data):
    Y = torch.cat([data.y for data in train_data])
    pos_ratio = Y.sum() / len(Y)
    pos_weight = (1 - pos_ratio) / pos_ratio
    return pos_weight


def compute_normalization_stats(train_data):
    """Compute z-score stats for features that vary across samples.

    Section properties are normalized per element type:
    - Columns (is_column == 1): I_col, A_col
    - Beams   (is_column == 0): I_beam, A_beam, span

    Load features are normalized globally (not split by element type)
    since they depend on node position, not element type.

    Other features are either binary, already bounded by construction,
    or spectrally bounded (Laplacian PE), so they don't need normalization.
    """
    X = torch.cat([data.x for data in train_data])
    is_column = X[:, FEATURE_NAMES.index("is_column")] == 1

    col_idx = feature_indices(NORM_COLUMN_FEATURES)
    beam_idx = feature_indices(NORM_BEAM_FEATURES)
    load_idx = feature_indices(NORM_LOAD_FEATURES)

    return {
        "col_mean": X[is_column][:, col_idx].mean(dim=0),
        "col_std": X[is_column][:, col_idx].std(dim=0),
        "beam_mean": X[~is_column][:, beam_idx].mean(dim=0),
        "beam_std": X[~is_column][:, beam_idx].std(dim=0),
        "load_mean": X[:, load_idx].mean(dim=0),
        "load_std": X[:, load_idx].std(dim=0),
    }


def normalize_data(data_list, stats):
    """Apply z-score normalization to section-size and load features in-place."""
    col_idx = feature_indices(NORM_COLUMN_FEATURES)
    beam_idx = feature_indices(NORM_BEAM_FEATURES)
    load_idx = feature_indices(NORM_LOAD_FEATURES)

    for data in data_list:
        is_column = data.x[:, FEATURE_NAMES.index("is_column")] == 1
        for i, ci in enumerate(col_idx):
            data.x[is_column, ci] = (data.x[is_column, ci] - stats["col_mean"][i]) / stats["col_std"][i]
        for i, bi in enumerate(beam_idx):
            data.x[~is_column, bi] = (data.x[~is_column, bi] - stats["beam_mean"][i]) / stats["beam_std"][i]
        for i, li in enumerate(load_idx):
            data.x[:, li] = (data.x[:, li] - stats["load_mean"][i]) / stats["load_std"][i]
    return data_list
