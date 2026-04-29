from pathlib import Path

import summarize_fourclient_full_avguard_fixed_lambda_anchor as base_summary


def config_path_for(stage3_dir):
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = base_summary.CONFIG_DIR_MAP.get(stage3_dir.parent.name)
    if not dataset_dir:
        return None
    return repo_root / "configs/avguard/fiveclient_full_avguard_fixed_lambda_anchor" / dataset_dir / "stage3_test.json"


base_summary.config_path_for = config_path_for


if __name__ == "__main__":
    base_summary.main()
