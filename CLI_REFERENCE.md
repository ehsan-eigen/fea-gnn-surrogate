# CLI Reference

## generate_dataset.py

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | Generation mode: `train` or `test` |
| `--datasets` | training defaults | Dataset keys from `config.json` (`dataset_1` through `dataset_6`) |
| `--num_samples` | `1000` | Number of structures to generate |
| `--config` | `config.json` | Path to the configuration file |
| `--output_dir` | `data` | Root output directory |
| `--output_label` | value of `--mode` | Subdirectory label under output_dir |
| `--output_name` | `pyg_line_graphs.pkl` | Filename for the saved PyG dataset |
| `--visualize` | off | Save a deflection plot for each structure |
| `--skip_fea` | off | Skip FEA (no labels — cannot train) |
| `--no_save_graphs` | off | Skip saving raw/simplified NetworkX graphs |
| `--concurrency` | `1` | Number of samples to generate in parallel |
| `--hf_repo` | none | Hugging Face dataset repo to upload to (reads `HF_TOKEN`) |

Always saves base NX line graphs to `data/{mode}/base/` alongside the PyG variants.

## extract_features.py

Re-extract PyG features from saved base NX line graphs. Use this when changing features — avoids re-running FEA.

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | Config section (must match a previous `generate_dataset.py` run) |
| `--data_dir` | `data` | Root data directory |
| `--output_name` | `pyg_line_graphs.pkl` | Filename for the saved PyG dataset |
| `--skip_fea` | off | Set if the base graphs have no labels (generated with `--skip_fea`) |

## train_surrogate.py

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | Data subdirectory for training data |
| `--model` | `mpnn` | Model architecture: `mpnn` (SharedMPNN) or `gps` (GPS Transformer) |
| `--edge_strategy` | `with_vn` | `with_vn` or `no_vn` |
| `--run_eval` | off | Evaluate on the pooled test set after training |
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
| `--save_path` | `best_model_<model>_<edge_strategy>.pth` | Model checkpoint path |
| `--log_dir` | `./logs/` | TensorBoard log directory |
| `--hf_repo` | none | Hugging Face model repo to upload trained model to (reads `HF_TOKEN`) |

## evaluate_model.py

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `best_model_<model>_<edge_strategy>.pth` | Path to trained model checkpoint; accepts `hf://owner/repo/file.pth` |
| `--model` | `mpnn` | Model architecture: `mpnn` or `gps` |
| `--test_sets` | (required) | Test set names (e.g. `test`) |
| `--edge_strategy` | `with_vn` | Edge strategy variant |
| `--data_dir` | `hf://ehsan94/fea-gnn-surrogate` | Root data directory or `hf://owner/repo` |
| `--hidden_dim` | `18` | Must match training |
| `--mp_steps` | `3` | Must match training |
| `--heads` | `3` | Must match training (GPS only) |
| `--dropout` | `0.2` | Must match training (GPS only) |
| `--attn_dropout` | `0.2` | Must match training (GPS only) |
| `--batch_size` | `32` | Batch size for evaluation |

## run_inference.py

| Argument | Default | Description |
|---|---|---|
| `--edge_strategy` | `with_vn` | Edge strategy variant |
| `--model` | `mpnn` | Model architecture: `mpnn` or `gps` |
| `--model_path` | `best_model_<model>_<edge_strategy>.pth` | Model checkpoint path |
| `--data_dir` | `data/test` | Data directory containing edge strategy subdirs |
| `--top_stability` | `15` | Top-K by predicted validity |
| `--top_weight` | `15` | Top-K lightest from those |
| `--output_dir` | `top_structs` | Deflection plot output directory |
| `--hidden_dim` | `18` | Must match training |
| `--mp_steps` | `3` | Must match training |
