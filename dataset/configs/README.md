# Config Layout

- `configs/avguard/<dataset>/...`
- The workspace keeps only the AVGuard staged pipeline configs.
- `configs/avguard/ieee_cis_fraud/` contains stage1/stage2/stage3 examples for the processed IEEE-CIS Fraud dataset.

LFBA configs use gradient-based poison-set inference by default. To run the supervised same-label variant, set `"lfba_poison_source": "oracle_label"` or use the `_oracle_label.json` examples where available.
