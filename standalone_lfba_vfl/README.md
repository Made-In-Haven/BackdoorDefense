# Standalone LFBA on Plain VFL

This folder extracts the LFBA attack path from the main project and keeps only:

- plain VFL training
- LFBA poisoning
- clean accuracy / target accuracy / ASR evaluation

It does not include the Anchor Defense workflow.

## Supported datasets

- `CIFAR10`
- `UCIHAR`
- `NUSWIDE`
- `PHISHING`

## Files

- `run.py`: standalone experiment entry
- `attack_core.py`: LFBA trigger injection and test-set rebuilding
- `trainer.py`: plain VFL training and ASR evaluation
- `configs/`: example configs for all four datasets

## Run

```bash
python -m standalone_lfba_vfl.run --config standalone_lfba_vfl/configs/cifar10_plain_vfl_lfba.json
```

```bash
python -m standalone_lfba_vfl.run --config standalone_lfba_vfl/configs/ucihar_plain_vfl_lfba.json
```

```bash
python -m standalone_lfba_vfl.run --config standalone_lfba_vfl/configs/ucihar_plain_vfl_lfba_test.json
```

```bash
python -m standalone_lfba_vfl.run --config standalone_lfba_vfl/configs/nuswide_plain_vfl_lfba.json
```

```bash
python -m standalone_lfba_vfl.run --config standalone_lfba_vfl/configs/nuswide_plain_vfl_lfba_test.json
```

```bash
python -m standalone_lfba_vfl.run --config standalone_lfba_vfl/configs/phishing_plain_vfl_lfba.json
```

```bash
python -m standalone_lfba_vfl.run --config standalone_lfba_vfl/configs/phishing_plain_vfl_lfba_test.json
```

## Notes

- `PHISHING` expects the updated file at `dataset/data_raw/Phishing/PHISHING_full.csv` and adapts to the CSV's feature dimension automatically.
- `UCIHAR` and `PHISHING` are now aligned to a comparable `client_num=3` setting, with the third client as attacker, the same `poison_rate=0.1`, and matched trigger-density budgets.
- `phishing_plain_vfl_lfba.json` is the no-defense training config, and `phishing_plain_vfl_lfba_test.json` reuses its best checkpoint for test-only evaluation.
- `ucihar_plain_vfl_lfba.json` is the no-defense training config, and `ucihar_plain_vfl_lfba_test.json` reuses its best checkpoint for test-only evaluation.
- The trainer rebuilds the poisoned test set before evaluation and reports `ASR`.
- `CIFAR10` uses an image patch trigger inside the attacker slice.
- `UCIHAR` uses fixed-value feature triggers on selected attacker dimensions.
- `NUSWIDE` and `PHISHING` use fixed binary-style vector triggers on attacker dimensions.
- `NUSWIDE` keeps the original image/text split when `client_num=2`; for `client_num>2`, the concatenated 1634-d feature vector is split into contiguous client slices.
- The best checkpoint is saved to `results_dir/best_checkpoint.pth.tar`.
