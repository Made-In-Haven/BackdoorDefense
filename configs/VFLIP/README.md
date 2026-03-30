# VFLIP Configs

This directory stores runnable config files for the standalone `VFLIP/VFLIP-esorics24/main.py` entry.

Available configs:

- `cifar10_project.json`
- `phishing_project.json`
- `nuswide_project.json`

Recommended commands:

```powershell
& 'D:\Code\Miniconda\envs\backdoor\python.exe' VFLIP\VFLIP-esorics24\main.py --config configs\VFLIP\cifar10_project.json
```

```powershell
& 'D:\Code\Miniconda\envs\backdoor\python.exe' VFLIP\VFLIP-esorics24\main.py --config configs\VFLIP\phishing_project.json
```

```powershell
& 'D:\Code\Miniconda\envs\backdoor\python.exe' VFLIP\VFLIP-esorics24\main.py --config configs\VFLIP\nuswide_project.json
```

Notes:

- These configs use `runtime = "project"` and the project checkpoints under `results/standalone_lfba_vfl/...`.
- The current default configs use `defense_type = "NONE"` because the bundled project checkpoints do not include VFLIP `MAE` weights.
- If you later have a checkpoint containing `mae`, `mae_mu`, `mae_std`, `s_score_mean`, and `s_score_std`, you can switch `defense_type` to `VFLIP`.
