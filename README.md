# MambaSleep-CVD

Pipeline for **long-term cardiovascular disease (CVD) survival prediction** from full-night polysomnography (PSG). The model fuses temporal and cross-modal dynamics from **ECG, EEG, and EOG** using a **Coupled Mamba** architecture.

This release provides unified scripts for **training** and **evaluation**.

## Overview

Two input pipelines are supported via a single `--mode` flag:

| Mode | Description | Input data |
|------|-------------|------------|
| `features` | Multimodal model on pre-extracted embeddings (USleep; **Self-DANA** for ECG @ 5 s — encoder release pending) | `.npy` feature files under `data_folder/features/` |
| `raw` | End-to-end model with built-in unimodal encoders | `.hdf5` raw signals under `data_folder/raw/` |

Both modes share the same dataset module, survival loss, and entry-point scripts.

## Repository structure

```
MambaSleep-CVD/
├── requirements.txt          # Python dependencies
├── README.md
├── checkpoints/              # Pre-trained Lightning .ckpt (see below)
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
│           ├── DANA/         # Self-DANA ECG @ 5 s (configs; encoder repo/weights TBD)
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

**Included in this repository:** pre-trained **checkpoints** under `checkpoints/` for **USleep** and **raw** Coupled Mamba runs (PyTorch Lightning `.ckpt`; layout below). **Self-DANA** encoder weights and the code repository to build ECG embeddings are **not yet publicly available**; configs under `scripts/sources/configs/DANA/` are provided for when that release is ready.

**Prepare locally:**

- PSG recordings and/or pre-extracted feature files from SHHS
- Outcome and demographic CSV files from https://sleepdata.org/datasets/shhs/files/datasets  
  Place the official NSRR files under `shhs_csv/` (at minimum `shhs-cvd-summary-dataset-0.21.0.csv`, `shhs-cvd-events-dataset-0.21.0.csv`, `shhs1-dataset-0.21.0.csv`, `shhs2-dataset-0.21.0.csv`).  
  Run `notebooks/build_shhs_outcome_from_cvd_summary.ipynb` to write `shhs_csv/results/shhs1_events_summary_from_summary_survival_not_filtered.csv` and `shhs2_events_summary_from_summary_survival_not_filtered_events.csv`. SHHS2 uses visit-relative times and post–visit-2 reconciliation from the CVD events file.
  `shhs1-dataset` / `shhs2-dataset` are the visit cohort tables (demographics, meds, …); the notebook also reads **`stdatep`** from `shhs2-dataset` to anchor SHHS2 follow-up times.

### Checkpoints

Weights live in `checkpoints/` (same path as `paths.checkpoint` in YAML). Each experiment folder contains **`best-checkpoint.ckpt`**.

Released runs use short folder names (e.g. `survival_all_demo`, `survival_no_demo`) under `features/USleep`, `features/DANA` or `raw`:

```
checkpoints/
├── features/
│   ├── USleep/          # ECG embeddings @ 1 s (USleep configs)
│   │   ├── survival_all_demo/
│   │   │   └── best-checkpoint.ckpt
│   │   └── survival_no_demo/
│   │       └── best-checkpoint.ckpt
│   └── DANA/            # ECG embeddings @ 5 s (DANA configs)
│       └── survival_all_demo/
│           └── best-checkpoint.ckpt
└── raw/
    └── survival_all_demo/
        └── best-checkpoint.ckpt
```

Use the **matching** global config for evaluation (folder `survival_no_demo` → `scripts/sources/configs/USleep/ablation/survival_from_summary_no_demo.yaml`, etc.). Pass **`--checkpoint_folder`** to the directory that directly contains the `.ckpt`, not the file path.


### Expected data layout

**USleep embeddings (`--mode features`, configs under `scripts/sources/configs/USleep/`).**  
Generate per-channel feature sequences with **[SLEEPYLAND](https://github.com/biomedical-signal-processing/sleepyland)** (U-Sleep encoder). Pre-trained weights are distributed via the [SLEEPYLAND Docker image](https://hub.docker.com/r/bspsupsi/sleepyland). Export **1 s** resolution embeddings and copy them into the tree below (channel names must match the YAML `model_config.modalities`). Typical USleep feature dimension is **6** per time step; arrays are stored as `(T, F)` `.npy` files.

**Self-DANA (ECG @ 5 s)** — configs in `scripts/sources/configs/DANA/` use ECG features at **`res_5s`** (see `resolution_input_embedding_s`). The **Self-DANA repository and pretrained encoder weights are not yet publicly available**, so new ECG embeddings for this branch cannot be generated from this release alone. When available, ECG files should follow `ECG/res_5s/ECG/shhs1-{subject_id}.npy` (EEG/EOG typically remain USleep **`res_1s`** as above).

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

Point `--checkpoint_folder` at a directory under `checkpoints/` that contains a `best*.ckpt` file (see [Checkpoints](#checkpoints)).

```bash
python scripts/eval.py \
  --mode features \
  --checkpoint_folder checkpoints/features/USleep/survival_all_demo \
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

If you use this code or the released checkpoints, please cite:

```bibtex
@article{MambaSleepCVD_2026,
  title   = {MambaSleepCVD for prediction of long term cardiovascular outcomes from polysomnography},
  author  = {Calzoni, Alessia and Dei Rossi, Alvise and Monachino, Giuliana and Zanchi, Beatrice and Fiorillo, Luigi and Savardi, Mattia and Faraci, Francesca Dalia and Signoroni, Alberto},
  journal = {npj Digital Medicine},
  year    = {2026},
  doi     = {10.1038/s41746-026-03450-3},
  url     = {https://doi.org/10.1038/s41746-026-03450-3}
}
```

See [`LICENSE`](LICENSE) for licensing terms.