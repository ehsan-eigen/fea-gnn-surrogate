import os
import torch
import numpy as np
import pandas as pd
import pickle

from fea_gnn_surrogate.surrogate.model import SharedMPNN, GPSModel
from fea_gnn_surrogate.surrogate.dataset import normalize_data
from fea_gnn_surrogate.graph.graph_utils import FEATURE_NAMES


def _resolve_model_path(model_path):
    """If model_path uses hf://owner/repo/filename, check locally first, then download."""
    if not model_path.startswith("hf://"):
        return model_path
    # hf://owner/repo-name/filename.pth  →  repo_id="owner/repo-name", filename="filename.pth"
    parts = model_path[len("hf://"):].split("/")
    if len(parts) < 3:
        raise ValueError(
            "hf:// paths must be hf://owner/repo-name/filename.pth"
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    # Check locally before attempting a network download
    for local_path in [filename, os.path.basename(filename)]:
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded model from hf://{repo_id}/{filename}")
    return local_path


def _build_model(model_type, num_features, hidden_dim, output_dim, num_layers,
                 heads=3, dropout=0.2, attn_dropout=0.2):
    """Instantiate a model by type name."""
    if model_type == "gps":
        return GPSModel(num_features, hidden_dim, output_dim, num_layers,
                        heads=heads, dropout=dropout, attn_dropout=attn_dropout)
    else:
        return SharedMPNN(num_features, hidden_dim, output_dim, num_layers)


def load_model(model_path, num_features, hidden_dim=18, output_dim=1,
               num_mp_steps=3, model_type="mpnn", heads=3,
               dropout=0.2, attn_dropout=0.2):
    """Load a trained model and normalization stats from a checkpoint.

    model_path can be a local path or hf://owner/repo-name/filename.pth.
    """
    checkpoint = torch.load(_resolve_model_path(model_path), weights_only=False)

    # Detect model type from checkpoint if saved there
    saved_type = None
    if "model_state_dict" in checkpoint:
        saved_type = checkpoint.get("model_type")
    if saved_type is not None:
        model_type = saved_type

    model = _build_model(model_type, num_features, hidden_dim, output_dim,
                         num_mp_steps, heads=heads, dropout=dropout,
                         attn_dropout=attn_dropout)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        norm_stats = checkpoint.get("norm_stats")
    else:
        # Legacy format: checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        norm_stats = None

    model.eval()
    return model, norm_stats


def predict(model, data_list, threshold=0.5):
    """Run inference on a list of PyG Data objects."""
    model.eval()
    results = []
    with torch.no_grad():
        for data in data_list:
            logits = model(data).view(-1)
            probs = torch.sigmoid(logits)
            mask = data.x[:, FEATURE_NAMES.index("real")] == 1
            probs = probs[mask]
            preds = (probs >= threshold).int()
            results.append({
                "probs": probs.numpy(),
                "preds": preds.numpy(),
                "min_prob": probs.min().item(),
                "all_valid": preds.all().item(),
                "weight": int(data.weight) if hasattr(data, "weight") else 0,
                "name": data.name if hasattr(data, "name") else None,
            })
    return results


def rank_structures(results, top_stability_count=15, top_weight_count=15):
    """Rank structures by validity prediction, then by weight."""
    S = np.zeros((len(results), 3))
    for i, r in enumerate(results):
        S[i, 0] = r["name"] if r["name"] is not None else i
        S[i, 1] = r["min_prob"]
        S[i, 2] = r["weight"]

    df = pd.DataFrame(data=S, columns=["name", "valid_ratio_prediction", "weight"])
    df["rank_prediction"] = df["valid_ratio_prediction"].rank(ascending=False, method="min")

    df = df.sort_values(by=["rank_prediction", "weight"], ascending=True).iloc[:top_stability_count]
    df = df.sort_values(by=["weight"], ascending=True).iloc[:top_weight_count]
    return df
