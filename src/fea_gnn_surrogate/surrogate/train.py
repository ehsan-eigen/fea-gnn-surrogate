import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from fea_gnn_surrogate.graph.graph_utils import FEATURE_NAMES

_REAL_IDX = FEATURE_NAMES.index("real")


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    predictions_list = []
    labels_list = []

    for batch_data in dataloader:
        optimizer.zero_grad()
        mask = batch_data.x[:, _REAL_IDX] == 1
        pred = model(batch_data).view(-1)[mask]
        labels = batch_data.y.to(torch.float64)[mask]
        loss = criterion(pred.to(torch.float64), labels)
        preds = torch.sigmoid(pred)

        loss.backward()
        optimizer.step()
        predictions_list.extend(preds.detach().numpy())
        labels_list.extend(labels.cpu().numpy())
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(labels_list, predictions_list)
    return avg_loss, auc


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    predictions_list = []
    labels_list = []
    names = []

    with torch.no_grad():
        for batch_data in dataloader:
            mask = batch_data.x[:, _REAL_IDX] == 1
            pred = model(batch_data).view(-1)[mask]
            labels = batch_data.y.to(torch.float64)[mask]
            loss = criterion(pred.to(torch.float64), labels)
            preds = torch.sigmoid(pred)
            predictions_list.extend(preds.detach().numpy())
            labels_list.extend(labels.cpu().numpy())
            total_loss += loss.item()
            if hasattr(batch_data, "name"):
                names.extend(batch_data.name)

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(labels_list, predictions_list)
    return avg_loss, auc, np.array(predictions_list), np.array(labels_list), names


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs,
          save_path="best_model.pth", log_dir="./logs/", norm_stats=None,
          hf_repo=None, model_type="mpnn", attn_bias=None):
    best_val_loss = float("inf")
    best_model_state_dict = None

    writer = SummaryWriter(log_dir + str(time.time()))

    for epoch in range(num_epochs):
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc, _, _, _ = validate(model, val_loader, criterion)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("AUC/Train", train_auc, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("AUC/Validation", val_auc, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs} — "
              f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} — "
              f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict().copy()

    writer.close()

    if best_model_state_dict is not None:
        checkpoint = {
            "model_state_dict": best_model_state_dict,
            "model_type": model_type,
        }
        if attn_bias is not None:
            checkpoint["attn_bias"] = attn_bias
        if norm_stats is not None:
            checkpoint["norm_stats"] = norm_stats
        torch.save(checkpoint, save_path)
        print(f"Best model saved to {save_path}")
        print(f"Best validation loss: {best_val_loss:.4f}")

        if hf_repo is not None:
            from huggingface_hub import upload_file
            filename = os.path.basename(save_path)
            upload_file(
                path_or_fileobj=save_path,
                path_in_repo=filename,
                repo_id=hf_repo,
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
            )
            print(f"Model uploaded to hf://{hf_repo}/{filename}")

    return best_model_state_dict
