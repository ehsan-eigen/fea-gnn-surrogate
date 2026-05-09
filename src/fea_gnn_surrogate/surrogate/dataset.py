import os
import pickle
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


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
    """Compute normalization stats from training data for column/beam features."""
    X = torch.cat([data.x for data in train_data])
    mask = X[:, 0] == 1
    mask_mean = X[mask, 8:10].mean(dim=0)
    mask_std = X[mask, 8:10].std(dim=0)
    not_mask_mean = X[~mask, 10:].mean(dim=0)
    not_mask_std = X[~mask, 10:].std(dim=0)
    return {
        "mask_mean": mask_mean,
        "mask_std": mask_std,
        "not_mask_mean": not_mask_mean,
        "not_mask_std": not_mask_std,
    }


def normalize_data(data_list, stats):
    """Apply normalization stats to a data list in-place."""
    mask_mean = stats["mask_mean"]
    mask_std = stats["mask_std"]
    not_mask_mean = stats["not_mask_mean"]
    not_mask_std = stats["not_mask_std"]

    for data in data_list:
        mask = data.x[:, 0] == 1
        data.x[mask, 8:10] = (data.x[mask, 8:10] - mask_mean) / mask_std
        data.x[~mask, 10:] = (data.x[~mask, 10:] - not_mask_mean) / not_mask_std
    return data_list
