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
python -m standalone_lfba_vfl.run --config standalone_lfba_vfl/configs/nuswide_plain_vfl_lfba.json
```

```bash
python -m standalone_lfba_vfl.run --config standalone_lfba_vfl/configs/phishing_plain_vfl_lfba.json
```

## Notes

- The trainer rebuilds the poisoned test set before evaluation and reports `ASR`.
- `CIFAR10` uses an image patch trigger inside the attacker slice.
- `UCIHAR` uses fixed-value feature triggers on selected attacker dimensions.
- `NUSWIDE` and `PHISHING` use fixed binary-style vector triggers on attacker dimensions.
- `NUSWIDE` currently supports `client_num=2` only because the original dataset split is image/text.
- The best checkpoint is saved to `results_dir/best_checkpoint.pth.tar`.
