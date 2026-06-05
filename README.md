# MambaSleep-CVD

Pipeline for **long-term cardiovascular disease (CVD) survival prediction** from full-night polysomnography (PSG). The model fuses temporal and cross-modal dynamics from **ECG, EEG, and EOG** using a **Coupled Mamba** architecture.

This release provides unified scripts for **training** and **evaluation**.

## Overview

Two input pipelines are supported via a single `--mode` flag:

| Mode | Description | Input data |
|------|-------------|------------|
| `features` | Multimodal model on pre-extracted embeddings (e.g. USleep, DANA) | `.npy` feature files under `data_folder/features/` |
| `raw` | End-to-end model with built-in unimodal encoders | `.hdf5` raw signals under `data_folder/raw/` |

Both modes share the same dataset module, survival loss, and entry-point scripts.

## Repository structure

```
MambaSleep-CVD/
├── requirements.txt          # Python dependencies
├── README.md
├── scripts/
│   ├── train.py              # Unified training script
│   ├── eval.py               # Unified evaluation script
│   └── sources/
│       ├── dataset.py        # Dataset & DataModule (features + raw)
│       ├── models/
│       │   ├── models.py           # Feature-based multimodal model
│       │   ├── models_raw.py       # Raw-signal multimodal model
│       │   └── efficient_coupled_mamba.py
│       ├── base_system/      # Loss, metrics, Lightning base class
│       ├── utils/            # CSV handling, splits, samplers
│       └── configs/          # Example YAML configs
│           ├── USleep/       # Feature configs (USleep embeddings)
│           ├── DANA/         # Feature configs (DANA embeddings)
│           └── raw/          # Raw-signal configs
```

## Requirements

- **Python** 3.10+
- **CUDA-enabled GPU** (recommended; Mamba CUDA kernels require Linux in practice)
- **PyTorch** with matching CUDA build
- See [`requirements.txt`](requirements.txt) for the full dependency list

### Installation

1. **Create a virtual environment** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

2. **Install PyTorch** first (choose the CUDA version for your system):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

3. **Install remaining dependencies**:

```bash
pip install -r requirements.txt
```

4. **Install Mamba packages** (required for the Coupled Mamba backbone):

```bash
pip install "mamba-ssm[causal-conv1d]>=2.0.0" --no-build-isolation
```

> On Windows, CUDA extensions for `mamba-ssm` may not build natively. Use **WSL2** or a **Linux** environment for training.

5. **Optional — Weights & Biases** (only if you use `--use_wandb` during training):

```bash
pip install wandb
```

## Data & checkpoints

The following are **not** distributed with this repository and must be prepared locally:

- PSG recordings and/or pre-extracted feature files from SHHS
- Outcome and demographic CSV files from https://sleepdata.org/datasets/shhs/files/datasets
- **Trained model checkpoints** (`.ckpt`)

### Expected data layout

**Features mode** (`--mode features`):

```
PSG_data/
└── features/
    ├── EEG/res_1s/C4-A1/shhs1-{subject_id}.npy
    ├── EOG/res_1s/EOG(L)-PG1/shhs1-{subject_id}.npy
    └── ECG/res_1s/ECG/shhs1-{subject_id}.npy
```

**Raw mode** (`--mode raw`):

```
PSG_data/
└── raw/
    └── shhs1-{subject_id}.hdf5
```

## Training

Run from the repository root:

```bash
python scripts/train.py \
  --mode features \
  --global_config scripts/sources/configs/USleep/survival_from_summary_all_demo.yaml \
  --task_config scripts/sources/configs/multilabel_task.yaml
```

Raw-signal training:

```bash
python scripts/train.py \
  --mode raw \
  --global_config scripts/sources/configs/raw/survival_all_demo.yaml \
  --task_config scripts/sources/configs/multilabel_task.yaml
```

### Main arguments

| Argument | Description |
|----------|-------------|
| `--mode` | `features` or `raw` |
| `--global_config` | Model, data, and training YAML |
| `--task_config` | Task-specific overrides (labels, WandB group, etc.) |
| `--only_demo_model` | Train demographic baseline only |
| `--use_wandb` | Enable Weights & Biases logging |
| `--imputing_method` | `remove_events`, `censor_date`, or `mask_events` (default: `mask_events`) |

Checkpoints are saved under `paths.checkpoint` as defined in the config (TensorBoard logs are written alongside them unless WandB is enabled).

## Evaluation

Evaluation requires a **local checkpoint folder** containing a `best-*.ckpt` file.

```bash
python scripts/eval.py \
  --mode features \
  --checkpoint_folder path/to/your/checkpoint/folder \
  --config scripts/sources/configs/USleep/survival_from_summary_all_demo.yaml \
  --task_config scripts/sources/configs/multilabel_task.yaml \
  --outcome_csv_file_path path/to/outcomes.csv \
  --demographic_csv_file_path path/to/demographics.csv \
  --dataset_split_path path/to/split.json \
  --dataset test \
  --output_dir evaluation_results
```

### Main arguments

| Argument | Description |
|----------|-------------|
| `--mode` | `features` or `raw` |
| `--checkpoint_folder` | Directory with `best-*.ckpt` |
| `--config` / `--task_config` | Same YAML structure as training |
| `--outcome_csv_file_path` | Outcomes CSV used for evaluation |
| `--demographic_csv_file_path` | Demographics CSV (optional) |
| `--dataset_split_path` | Split JSON (overrides config `split_path`) |
| `--dataset` | `test`, `val`, `train`, `primary_prev`, or `secondary_prev` |
| `--auroc_years` | Time points (years) for time-dependent AUROC |
| `--n_bootstraps` | Bootstrap iterations for confidence intervals |
| `--save_predictions` | Save predictions to `.npz` |

Results are written to `--output_dir` as JSON and CSV files.

## Configuration

Configs are split into two layers:

1. **Global config** (`survival_*.yaml`) — paths, modalities, model hyperparameters, training settings
2. **Task config** (`multilabel_task.yaml`) — target outcomes, imputation strategy, WandB group name

Ablation configs under `scripts/sources/configs/*/ablation/` vary modalities and demographic features.

## Citation & license

If you use this code, please cite the associated publication.

See [`LICENSE`](LICENSE) for licensing terms.