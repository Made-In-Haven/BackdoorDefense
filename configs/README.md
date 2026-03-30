# Config Layout

Recommended structure:

- `configs/avguard/<dataset>/...`
  - configs for the `AVGuard` defense pipeline
- `configs/baselines/<dataset>/...`
  - attack-only or no-defense baselines
- `configs/VFLIP/...`
  - runnable configs for the standalone `VFLIP/VFLIP-esorics24/main.py` entry on project datasets

Legacy root-level config files are kept for backward compatibility.
New experiments should prefer the organized paths below.
