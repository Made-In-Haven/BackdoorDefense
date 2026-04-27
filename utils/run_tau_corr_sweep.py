import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path


NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
TEST_EPOCH_RE = re.compile(
    rf"=> Test Epoch: (?P<epoch>\d+), .*? clean acc: (?P<clean_acc>{NUMBER_RE}), "
    rf"Top-(?P<top_k>\d+): (?P<topk_acc>{NUMBER_RE}), ASR: (?P<asr>{NUMBER_RE}), "
    rf"RAC: (?P<rac>{NUMBER_RE}), test target accuracy: (?P<test_target_accuracy>{NUMBER_RE}), "
    rf"stage3 final accuracy: (?P<stage3_final_acc>{NUMBER_RE}), "
    rf"stage3 final target accuracy: (?P<stage3_final_target_accuracy>{NUMBER_RE}), "
    rf"stage3 final asr: (?P<stage3_final_asr>{NUMBER_RE}), anchor loss: (?P<anchor_loss>{NUMBER_RE})"
)
DETECTION_RE = re.compile(
    rf"=> Stage 3 Detection Summary: recall: (?P<detection_recall>{NUMBER_RE}), "
    rf"precision: (?P<detection_precision>{NUMBER_RE}), f1: (?P<detection_f1>{NUMBER_RE}), "
    rf"false positive rate: (?P<false_positive_rate>{NUMBER_RE}), correction rate: (?P<correction_rate>{NUMBER_RE})"
)
CLEAN_DEBUG_RE = re.compile(
    r"=> Stage 3 Debug \(clean\): suspicious=(?P<clean_suspicious>\d+)/(?P<clean_total>\d+), "
    r"correction_applied=(?P<clean_correction_applied>\d+), "
    r"prediction_changed=(?P<clean_prediction_changed>\d+), "
    r"skipped_tied_vote=(?P<clean_skipped_tied_vote>\d+), "
    r"skipped_low_margin=(?P<clean_skipped_low_margin>\d+), "
    r"skipped_other=(?P<clean_skipped_other>\d+)"
)
POISON_DEBUG_RE = re.compile(
    r"=> Stage 3 Debug \(poison-valid\): suspicious=(?P<poison_valid_suspicious>\d+)/(?P<poison_valid_total>\d+), "
    r"correction_applied=(?P<poison_valid_correction_applied>\d+), "
    r"prediction_changed=(?P<poison_valid_prediction_changed>\d+), "
    r"skipped_tied_vote=(?P<poison_valid_skipped_tied_vote>\d+), "
    r"skipped_low_margin=(?P<poison_valid_skipped_low_margin>\d+), "
    r"skipped_other=(?P<poison_valid_skipped_other>\d+)"
)
ATTACK_DEBUG_RE = re.compile(
    r"=> Stage 3 Debug \(attack-success\): suspicious=(?P<attack_success_suspicious>\d+)/(?P<attack_success_total>\d+), "
    r"correction_applied=(?P<attack_success_correction_applied>\d+), "
    r"prediction_changed=(?P<attack_success_prediction_changed>\d+), "
    r"corrected_to_true=(?P<attack_success_corrected_to_true>\d+)"
)

DATASETS = {
    "CIFAR10": {
        "config_dir": "cifar10",
        "result_dir": "CIFAR10",
    },
    "NUSWIDE": {
        "config_dir": "nuswide",
        "result_dir": "NUSWIDE",
    },
    "IEEE_CIS_FRAUD": {
        "config_dir": "ieee_cis_fraud",
        "result_dir": "IEEE_CIS_FRAUD",
    },
    "PHISHING": {
        "config_dir": "phishing",
        "result_dir": "PHISHING",
    },
}

DEFAULT_TAU_CORRS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
SUMMARY_FIELDS = [
    "dataset",
    "tau_corr",
    "status",
    "return_code",
    "is_baseline",
    "is_selected_best",
    "selection_reason",
    "epoch",
    "clean_acc",
    "asr",
    "rac",
    "detection_recall",
    "detection_precision",
    "detection_f1",
    "false_positive_rate",
    "correction_rate",
    "clean_suspicious",
    "clean_total",
    "clean_correction_applied",
    "clean_prediction_changed",
    "poison_valid_suspicious",
    "poison_valid_total",
    "poison_valid_correction_applied",
    "poison_valid_prediction_changed",
    "attack_success_suspicious",
    "attack_success_total",
    "attack_success_correction_applied",
    "attack_success_prediction_changed",
    "attack_success_corrected_to_true",
    "results_dir",
    "log_path",
]


def repo_root():
    return Path(__file__).resolve().parents[1]


def tau_dir_name(tau_corr):
    return "tau_{:.2f}".format(float(tau_corr))


def read_json_with_comments(path):
    raw_text = path.read_text(encoding="utf-8-sig")
    output = []
    in_string = False
    escape = False
    index = 0
    while index < len(raw_text):
        char = raw_text[index]
        next_char = raw_text[index + 1] if index + 1 < len(raw_text) else ""
        if char == '"' and not escape:
            in_string = not in_string
        if not in_string and char == "/" and next_char == "/":
            while index < len(raw_text) and raw_text[index] != "\n":
                index += 1
            continue
        output.append(char)
        escape = char == "\\" and not escape
        if char != "\\":
            escape = False
        index += 1
    return json.loads("".join(output))


def maybe_float(value):
    if value in ("", None):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def parse_log(log_path):
    row = {}
    if not log_path.exists():
        return row

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        matchers = (TEST_EPOCH_RE, DETECTION_RE, CLEAN_DEBUG_RE, POISON_DEBUG_RE, ATTACK_DEBUG_RE)
        for regex in matchers:
            match = regex.search(line)
            if match:
                row.update(match.groupdict())
                break
    return row


def has_completed_log(log_path):
    parsed = parse_log(log_path)
    return "clean_acc" in parsed and "asr" in parsed


def build_command(args, dataset_name, tau_corr, results_dir, config_path, checkpoint_path):
    return [
        args.python,
        str(args.main),
        "--config",
        str(config_path),
        "--mode",
        "test",
        "--enable_conservative_correction",
        "true",
        "--tau_corr",
        "{:.2f}".format(float(tau_corr)),
        "--test_checkpoint",
        str(checkpoint_path),
        "--results_dir",
        str(results_dir),
    ]


def command_to_text(command):
    return subprocess.list2cmdline([str(part) for part in command])


def write_command_file(results_dir, command):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "run_command.txt").write_text(command_to_text(command) + "\n", encoding="utf-8")


def run_one(args, dataset_name, tau_corr, config_path, checkpoint_path, results_dir):
    log_path = results_dir / "experiment.log"
    command = build_command(args, dataset_name, tau_corr, results_dir, config_path, checkpoint_path)

    if args.dry_run:
        print(command_to_text(command))
        return "dry_run", None

    write_command_file(results_dir, command)

    if args.summarize_only:
        return ("parsed" if has_completed_log(log_path) else "missing"), None

    if log_path.exists() and not args.rerun and has_completed_log(log_path):
        print("Skip existing completed run: {} {}".format(dataset_name, tau_dir_name(tau_corr)))
        return "skipped", None

    print("Run {} {}".format(dataset_name, tau_dir_name(tau_corr)))
    completed = subprocess.run(command, cwd=args.repo_root)
    return ("completed" if completed.returncode == 0 else "failed"), completed.returncode


def numeric_row_value(row, key):
    return maybe_float(row.get(key))


def satisfies_constraints(row, baseline):
    recall = numeric_row_value(row, "detection_recall")
    correction = numeric_row_value(row, "correction_rate")
    asr = numeric_row_value(row, "asr")
    base_recall = numeric_row_value(baseline, "detection_recall")
    base_correction = numeric_row_value(baseline, "correction_rate")
    base_asr = numeric_row_value(baseline, "asr")
    if None in (recall, correction, asr, base_recall, base_correction, base_asr):
        return False
    return (
        recall >= base_recall - 0.01
        and correction >= base_correction - 0.03
        and asr <= base_asr + 0.02
    )


def dominates(left, right):
    left_clean = numeric_row_value(left, "clean_acc")
    right_clean = numeric_row_value(right, "clean_acc")
    left_asr = numeric_row_value(left, "asr")
    right_asr = numeric_row_value(right, "asr")
    left_recall = numeric_row_value(left, "detection_recall")
    right_recall = numeric_row_value(right, "detection_recall")
    left_correction = numeric_row_value(left, "correction_rate")
    right_correction = numeric_row_value(right, "correction_rate")
    if None in (left_clean, right_clean, left_asr, right_asr, left_recall, right_recall, left_correction, right_correction):
        return False

    no_worse = (
        left_clean >= right_clean
        and left_asr <= right_asr
        and left_recall >= right_recall
        and left_correction >= right_correction
    )
    strictly_better = (
        left_clean > right_clean
        or left_asr < right_asr
        or left_recall > right_recall
        or left_correction > right_correction
    )
    return no_worse and strictly_better


def select_best(rows):
    valid_rows = [row for row in rows if row.get("status") in {"completed", "skipped", "parsed"} and row.get("clean_acc")]
    if not valid_rows:
        return None, "no_valid_rows"

    baseline_rows = [row for row in valid_rows if abs(float(row["tau_corr"]) - 0.0) < 1e-12]
    if not baseline_rows:
        return None, "missing_baseline"
    baseline = baseline_rows[0]

    constrained = [row for row in valid_rows if satisfies_constraints(row, baseline)]
    if constrained:
        return max(constrained, key=lambda row: numeric_row_value(row, "clean_acc") or float("-inf")), "constraints_satisfied"

    pareto_rows = []
    for row in valid_rows:
        if not any(dominates(other, row) for other in valid_rows if other is not row):
            pareto_rows.append(row)
    if not pareto_rows:
        return baseline, "fallback_baseline"
    return max(
        pareto_rows,
        key=lambda row: (
            numeric_row_value(row, "clean_acc") or float("-inf"),
            -(numeric_row_value(row, "asr") or float("inf")),
        ),
    ), "pareto_fallback"


def as_csv_value(value):
    if value is None:
        return ""
    numeric = maybe_float(value)
    if numeric is None:
        return str(value)
    return "{:.6g}".format(numeric)


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: as_csv_value(row.get(field, "")) for field in SUMMARY_FIELDS})


def dataset_paths(args, dataset_name):
    spec = DATASETS[dataset_name]
    config_path = args.config_root / spec["config_dir"] / "stage3_test.json"
    checkpoint_path = args.result_root / spec["result_dir"] / "stage2" / "best_checkpoint.pth.tar"
    sweep_root = args.result_root / spec["result_dir"] / "stage3_tau_corr_sweep"
    return config_path, checkpoint_path, sweep_root


def build_row(dataset_name, tau_corr, results_dir, status, return_code):
    log_path = results_dir / "experiment.log"
    row = {
        "dataset": dataset_name,
        "tau_corr": "{:.2f}".format(float(tau_corr)),
        "status": status,
        "return_code": return_code,
        "is_baseline": "1" if abs(float(tau_corr)) < 1e-12 else "0",
        "results_dir": results_dir.as_posix(),
        "log_path": log_path.as_posix(),
    }
    row.update(parse_log(log_path))
    return row


def run_dataset(args, dataset_name):
    config_path, checkpoint_path, sweep_root = dataset_paths(args, dataset_name)
    if not config_path.exists():
        raise FileNotFoundError("Missing config for {}: {}".format(dataset_name, config_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError("Missing checkpoint for {}: {}".format(dataset_name, checkpoint_path))

    config_snapshot = read_json_with_comments(config_path)
    if not args.dry_run:
        sweep_root.mkdir(parents=True, exist_ok=True)
        (sweep_root / "base_config_snapshot.json").write_text(
            json.dumps(config_snapshot, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    rows = []
    for tau_corr in args.tau_corrs:
        results_dir = sweep_root / tau_dir_name(tau_corr)
        status, return_code = run_one(args, dataset_name, tau_corr, config_path, checkpoint_path, results_dir)
        rows.append(build_row(dataset_name, tau_corr, results_dir, status, return_code))

    best_row, selection_reason = select_best(rows)
    if best_row is not None:
        for row in rows:
            is_best = row["dataset"] == best_row["dataset"] and row["tau_corr"] == best_row["tau_corr"]
            row["is_selected_best"] = "1" if is_best else "0"
            row["selection_reason"] = selection_reason if is_best else ""
    else:
        for row in rows:
            row["is_selected_best"] = "0"
            row["selection_reason"] = selection_reason

    if not args.dry_run:
        write_summary(sweep_root / "summary.csv", rows)
    return rows


def parse_tau_corrs(raw_values):
    tau_corrs = []
    for raw_value in raw_values:
        for item in str(raw_value).split(","):
            stripped = item.strip()
            if stripped:
                tau_corrs.append(float(stripped))
    return tau_corrs


def build_parser():
    parser = argparse.ArgumentParser(description="Run Stage 3 tau_corr margin-gate sweeps.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--tau-corrs", nargs="+", default=[str(value) for value in DEFAULT_TAU_CORRS])
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs/avguard/fourclient_full_avguard_fixed_lambda_anchor"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor"),
    )
    parser.add_argument("--main", type=Path, default=Path("main.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rerun", action="store_true", help="rerun even if a parseable experiment.log exists")
    parser.add_argument("--dry-run", action="store_true", help="print commands and write command files without running")
    parser.add_argument("--summarize-only", action="store_true", help="only parse existing logs and rewrite summaries")
    return parser


def normalize_args(args):
    args.repo_root = args.repo_root.resolve()
    args.config_root = (args.repo_root / args.config_root).resolve() if not args.config_root.is_absolute() else args.config_root
    fallback_config_root = args.repo_root / "dataset" / "configs" / "avguard" / "fourclient_full_avguard_fixed_lambda_anchor"
    if not args.config_root.exists() and fallback_config_root.exists():
        args.config_root = fallback_config_root.resolve()
    args.result_root = (args.repo_root / args.result_root).resolve() if not args.result_root.is_absolute() else args.result_root
    args.main = (args.repo_root / args.main).resolve() if not args.main.is_absolute() else args.main
    args.tau_corrs = parse_tau_corrs(args.tau_corrs)
    if not args.tau_corrs:
        raise ValueError("At least one tau_corr value is required.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    normalize_args(args)

    all_rows = []
    for dataset_name in args.datasets:
        all_rows.extend(run_dataset(args, dataset_name))

    if not args.dry_run:
        global_summary_path = args.result_root / "stage3_tau_corr_sweep_summary.csv"
        write_summary(global_summary_path, all_rows)
        print("Wrote {}".format(global_summary_path))


if __name__ == "__main__":
    main()
