# BackdoorDefense: LFBA Attack and AVGuard Defense in Vertical Federated Learning

This repository implements **vertical federated learning (VFL)** experiments with the **LFBA** (label-free backdoor) attack and the **AVGuard** anchor-based defense: three-stage training (anchor pretraining, joint training with anchor loss / EMA calibration, and test-time detection with joint weighted correction). The integrated entry point is `main.py` with JSON configs under `configs/avguard/`.

![](./framework.svg)

## Background (LFBA)

> [Label-Free Backdoor Attacks in Vertical Federated Learning](https://ojs.aaai.org/index.php/AAAI/article/view/34246)  
> Wei Shen, Wenke Huang, Guancheng Wan, Mang Ye — Wuhan University

VFL trains a global model from distributed features and shared sample IDs. **LFBA** constructs a poison set without extra label knowledge by using embedding-gradient cues and selective sample switching so the backdoor trigger is learned together with clean accuracy.

## Requirements

- Python 3.9+ recommended  
- PyTorch / CUDA as in `requirements.txt` (pinned around `torch==1.12.1`). If CUDA is unavailable, `main.py` falls back to CPU with a log message.  
- A GPU (e.g. single RTX 3090 class card) matches the original LFBA evaluation setup; CPU runs are possible but slow.

Install from the repository root:

```bash
conda create -n backdoor_defense python=3.9
conda activate backdoor_defense
cd BackdoorDefense   # or your clone path
pip install -r requirements.txt
```

## Repository layout

| Path | Role |
|------|------|
| `main.py` | CLI, config merge, data/models, defense + attack runtime, training / testing |
| `attack/` | LFBA trigger and poisoning runtime |
| `defense/` | AVGuard anchor defense, detector, trainer hooks |
| `dataset/` | VFL dataset loaders (`dataset/README.md` documents NUS-WIDE feature layout) |
| `models/` | Global / local models per dataset |
| `configs/avguard/` | Staged JSON configs (per-dataset subfolders and ablation grids) |
| `configs/README.md` | Config layout and `oracle_label` vs gradient poison source |
| `utils/` | Training loop, seeds, helpers |
| `draw_pictures/` | Optional plotting scripts |
| `results/` | Run outputs; Git tracks `*.csv` / `*.log` only (see `.gitignore`; checkpoints `*.pt` / `*.pth.tar` stay local) |

## Datasets and `data_dir`

Default CLI `--data_dir` is `dataset/`. JSON configs often set `"data_dir": "./dataset/data_raw/"`. Resolve paths relative to that root.

- **CIFAR10** — Torchvision CIFAR-10 under `data_dir` (`download=True` in code if missing).
- **PHISHING** — File `PHISHING_full.csv` in one of: `data_dir/Phishing/`, `data_dir/data_raw/Phishing/`, or `data_dir/` (root). Label column: `phishing` or `Result`.
- **IEEE_CIS_FRAUD** — Directory `IEEE-CIS-Fraud` containing `X_balanced.npy`, `y_balanced.npy`, and optionally `feature_columns.csv`, under one of: `data_dir/processed/`, `data_dir/data_raw/processed/`, or `data_dir/`.
- **NUSWIDE** — Under `data_dir`, expected layout includes `Groundtruth/TrainTestLabels/` and `Low_Level_Features/` (see `dataset/utils.py` and `dataset/README.md`). Supported `client_num`: 2, 3, 4, or 5.
- **UCIHAR** — UCI HAR features under `data_dir` per `UCIHAR_VFL` loader.

Dataset name aliases such as `IEEE-CIS-Fraud` / `IEEECISFRAUD` are canonicalized to `IEEE_CIS_FRAUD` at runtime.

## AVGuard workflow

Use staged configs (examples below; swap dataset folder as needed):

1. **Stage 1 — anchor pretraining** (`mode`: `pretrain_anchor`)

```bash
python main.py --config configs/avguard/phishing/stage1.json
```

2. **Stage 2 — joint training** with anchor loss and EMA anchor calibration (`mode`: `train`)

```bash
python main.py --config configs/avguard/phishing/stage2.json
```

3. **Stage 3 — evaluation** with support detection and joint weighted voting (`mode`: `test`)

```bash
python main.py --config configs/avguard/phishing/stage3_test.json
```

Other datasets:

- CIFAR-10: `configs/avguard/cifar10/` (e.g. `stage1_3clients.json`, `stage2_3clients.json`, …)  
- NUS-WIDE: `configs/avguard/nuswide/`  
- IEEE-CIS Fraud: `configs/avguard/ieee_cis_fraud/`  
- Ablations (e.g. gamma, static reliability, 4 clients): `configs/avguard/stage3_*`, `configs/avguard/phishing/stage2_noema.json`, etc.

Configs can set `"lfba_poison_source": "oracle_label"` or use the `*_oracle_label.json` templates for the supervised same-label poison pool (see `configs/README.md`).

## Useful CLI / config fields

| Field | Meaning |
|-------|--------|
| `--config` | JSON path; values override argparse defaults |
| `--device` | GPU index (invalid index falls back to `cuda:0` if CUDA exists) |
| `--dataset` | `CIFAR10`, `NUSWIDE`, `UCIHAR`, `PHISHING`, `IEEE_CIS_FRAUD` |
| `--mode` | `pretrain_anchor`, `train`, or `test` |
| `--results_dir` | If empty, defaults to `results/<DATASET>/<timestamp>/` |
| `--resume_latest` | Resume from `results_dir/latest_checkpoint.pth.tar` when training |
| `--attack` | e.g. `LFBA` |
| `--anchor_idx` | Anchor sample index; LFBA infers target label from this sample |
| `--poison_rate`, `--poison_dimensions`, `--select_rate` | LFBA poisoning and switching |
| `--lfba_poison_source` | `gradient` (label-free) or `oracle_label` |
| `--theta_supp` | Stage 3 valid-support threshold on local confidence / agreement |
| `--gamma` | Stage 3 joint voting temperature |
| `stage3_confidence_mode` | `raw_ratio` or `bounded_relative_gap` (see config JSONs) |

Training and metrics are written under `results_dir` (`experiment.log`, checkpoints, JSON metrics). Stage 2/3 configs should point `anchor_stage1_dir` / checkpoints to the Stage 1 output you intend to use.

## Citation

If you use the LFBA attack formulation, please cite:

```text
@inproceedings{shen2025label,
  title={Label-free backdoor attacks in vertical federated learning},
  author={Shen, Wei and Huang, Wenke and Wan, Guancheng and Ye, Mang},
  booktitle={The 39th AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

LFBA authors: [weishen@whu.edu.cn](mailto:weishen@whu.edu.cn)

For the AVGuard implementation and experiment scripts in this fork/workspace, cite or acknowledge this repository as appropriate for your publication context.
