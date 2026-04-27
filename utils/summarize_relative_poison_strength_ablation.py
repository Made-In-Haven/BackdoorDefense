import argparse
import csv
import json
import math
import re
from pathlib import Path


NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
COMBO_RE = re.compile(r"eta_(?P<eta>[^_]+)_poison_(?P<poison_rate>.+)")
TEST_EPOCH_RE = re.compile(
    rf"=> Test Epoch: (?P<epoch>\d+), .*? clean acc: (?P<clean_accuracy>{NUMBER_RE}), "
    rf"Top-(?P<top_k>\d+): (?P<topk_accuracy>{NUMBER_RE}), ASR: (?P<attack_success_rate>{NUMBER_RE}), "
    rf"RAC: (?P<rac>{NUMBER_RE}), test target accuracy: (?P<test_target_accuracy>{NUMBER_RE}), "
    rf"stage3 final accuracy: (?P<stage3_final_accuracy>{NUMBER_RE}), "
    rf"stage3 final target accuracy: (?P<stage3_final_target_accuracy>{NUMBER_RE}), "
    rf"stage3 final asr: (?P<defense_asr>{NUMBER_RE}), anchor loss: (?P<anchor_loss>{NUMBER_RE})"
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

FIELDNAMES = [
    "dataset",
    "eta",
    "poison_rate",
    "lambda_stage1_anchor",
    "anchor_margin",
    "anchor_scale",
    "anchor_ema_momentum",
    "clean_accuracy",
    "attack_success_rate",
    "defense_asr",
    "corrected_asr",
    "stage3_final_accuracy",
    "stage3_final_target_accuracy",
    "detection_recall",
    "detection_precision",
    "detection_f1",
    "false_positive_rate",
    "correction_rate",
    "clean_suspicious",
    "clean_total",
    "clean_correction_applied",
    "poison_valid_suspicious",
    "poison_valid_total",
    "poison_valid_correction_applied",
    "attack_success_suspicious",
    "attack_success_total",
    "attack_success_correction_applied",
    "attack_success_corrected_to_true",
    "valid_support_rate",
    "support_detection_rate",
    "avg_valid_support_count",
    "epoch",
    "top_k",
    "topk_accuracy",
    "rac",
    "test_target_accuracy",
    "anchor_loss",
    "stage3_result_directory",
]


def strip_json_line_comments(raw_text):
    result = []
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

        result.append(char)
        escape = char == "\\" and not escape
        if char != "\\":
            escape = False
        index += 1
    return "".join(result)


def load_json(path):
    return json.loads(strip_json_line_comments(path.read_text(encoding="utf-8-sig")))


def combo_values(combo_name):
    match = COMBO_RE.fullmatch(combo_name)
    if not match:
        return "", ""
    return match.group("eta"), match.group("poison_rate")


def blank_row(stage3_dir):
    eta, poison_rate = combo_values(stage3_dir.parent.name)
    return {
        "dataset": stage3_dir.parent.parent.name,
        "eta": eta,
        "poison_rate": poison_rate,
        "stage3_result_directory": stage3_dir.as_posix(),
    }


def as_number_string(value):
    if value in ("", None):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.6g}"


def merge_match(row, match):
    row.update(match.groupdict())


def parse_log(log_path):
    row = {}
    if not log_path.exists():
        return row

    pending_test_row = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        test_match = TEST_EPOCH_RE.search(line)
        if test_match:
            pending_test_row = test_match.groupdict()
            row.update(pending_test_row)
            row["corrected_asr"] = pending_test_row.get("defense_asr", "")
            continue

        detection_match = DETECTION_RE.search(line)
        if detection_match:
            merge_match(row, detection_match)
            continue

        clean_match = CLEAN_DEBUG_RE.search(line)
        if clean_match:
            merge_match(row, clean_match)
            continue

        poison_match = POISON_DEBUG_RE.search(line)
        if poison_match:
            merge_match(row, poison_match)
            continue

        attack_match = ATTACK_DEBUG_RE.search(line)
        if attack_match:
            merge_match(row, attack_match)

    return row


def find_metric_files(stage3_dir):
    names = {
        "metrics.json",
        "stage3_metrics.json",
        "test_metrics.json",
        "result.json",
        "results.json",
    }
    return sorted(path for path in stage3_dir.rglob("*.json") if path.name in names)


def flatten_json(prefix, value, output):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_json(child_prefix, child, output)
    else:
        output[prefix] = value


def first_present(flat, candidates):
    for key in candidates:
        if key in flat and flat[key] not in ("", None):
            return flat[key]
    return ""


def parse_metric_files(stage3_dir):
    merged = {}
    mappings = {
        "clean_accuracy": ["clean_accuracy", "clean_acc", "accuracy", "metrics.clean_acc"],
        "attack_success_rate": ["attack_success_rate", "asr", "test_asr", "metrics.asr"],
        "defense_asr": ["defense_asr", "stage3_final_asr", "metrics.stage3_final_asr"],
        "corrected_asr": ["corrected_asr", "stage3_final_asr", "metrics.stage3_final_asr"],
        "detection_recall": ["detection_recall", "detection_rate", "metrics.detection_recall"],
        "detection_precision": ["detection_precision", "metrics.detection_precision"],
        "detection_f1": ["detection_f1", "metrics.detection_f1"],
        "false_positive_rate": ["false_positive_rate", "metrics.false_positive_rate"],
        "correction_rate": ["correction_rate", "metrics.correction_rate"],
        "valid_support_rate": ["valid_support_rate", "metrics.valid_support_rate"],
        "support_detection_rate": ["support_detection_rate", "metrics.support_detection_rate"],
        "avg_valid_support_count": ["avg_valid_support_count", "metrics.avg_valid_support_count"],
    }
    for metric_file in find_metric_files(stage3_dir):
        try:
            flat = {}
            flatten_json("", load_json(metric_file), flat)
        except (OSError, json.JSONDecodeError):
            continue

        for output_key, candidates in mappings.items():
            if output_key not in merged or merged[output_key] == "":
                merged[output_key] = first_present(flat, candidates)
    return merged


def config_path_for(stage3_dir):
    repo_root = Path(__file__).resolve().parents[1]
    dataset = stage3_dir.parent.parent.name.lower()
    combo_name = stage3_dir.parent.name
    return (
        repo_root
        / "configs"
        / "avguard"
        / "relative_poison_strength_ablation"
        / dataset
        / combo_name
        / "stage3_test.json"
    )


def parse_config(stage3_dir):
    path = config_path_for(stage3_dir)
    if not path.exists():
        return {}
    try:
        config = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "dataset": config.get("dataset", ""),
        "poison_rate": config.get("poison_rate", ""),
        "lambda_stage1_anchor": config.get("lambda_stage1_anchor", ""),
        "anchor_margin": config.get("anchor_margin", ""),
        "anchor_scale": config.get("anchor_scale", ""),
        "anchor_ema_momentum": config.get("anchor_ema_momentum", ""),
    }


def find_stage3_dirs(root):
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*/eta_*_poison_*/stage3") if path.is_dir())


def sort_key(row):
    dataset = str(row.get("dataset", ""))
    try:
        eta = float(row.get("eta", "inf"))
    except ValueError:
        eta = math.inf
    try:
        poison_rate = float(row.get("poison_rate", "inf"))
    except ValueError:
        poison_rate = math.inf
    return dataset, eta, poison_rate


def build_rows(root):
    rows = []
    for stage3_dir in find_stage3_dirs(root):
        row = blank_row(stage3_dir)
        row.update(parse_config(stage3_dir))
        row.update(parse_metric_files(stage3_dir))
        row.update(parse_log(stage3_dir / "experiment.log"))
        rows.append(row)
    return sorted(rows, key=sort_key)


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: as_number_string(row.get(key, "")) for key in FIELDNAMES})


def main():
    parser = argparse.ArgumentParser(description="Summarize AVGuard relative poison strength Stage 3 results.")
    parser.add_argument("--root", required=True, help="Root result directory to scan")
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    rows = build_rows(Path(args.root))
    write_csv(rows, Path(args.output))
    print(f"Wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
