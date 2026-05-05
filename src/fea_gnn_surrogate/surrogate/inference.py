import torch
import numpy as np
import pandas as pd
import pickle

from fea_gnn_surrogate.surrogate.model import SharedMPNN
from fea_gnn_surrogate.surrogate.dataset import normalize_data


def load_model(model_path, num_features, hidden_dim=18, output_dim=1, num_mp_steps=3):
    """Load a trained model and normalization stats from a checkpoint."""
    checkpoint = torch.load(model_path, weights_only=False)

    model = SharedMPNN(num_features, hidden_dim, output_dim, num_mp_steps)

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
            mask = data.x[:, 4] == 1  # exclude virtual node (real=0)
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
