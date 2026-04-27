import argparse
import csv
import json
import math
import re
from pathlib import Path


NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
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

LAMBDA_ANCHOR_FIELDS = [
    "dataset",
    "lambda_anchor",
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
    "epoch",
    "top_k",
    "topk_accuracy",
    "rac",
    "test_target_accuracy",
    "anchor_loss",
    "stage3_result_directory",
]

LAMBDA_STAGE1_FIELDS = [
    "dataset",
    "lambda_stage1_anchor",
    "anchor_margin",
    "anchor_scale",
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
    "epoch",
    "top_k",
    "topk_accuracy",
    "rac",
    "test_target_accuracy",
    "anchor_loss",
    "stage3_result_directory",
]

STAGE1_ABLATION_FIELDS = [
    "dataset",
    "ablation_variant",
    "enable_stage1",
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


def as_cell(value):
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.6g}"


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
    names = {"metrics.json", "stage3_metrics.json", "test_metrics.json", "result.json", "results.json"}
    row = {}
    for metric_file in sorted(path for path in stage3_dir.rglob("*.json") if path.name in names):
        try:
            flat = {}
            flatten_json("", load_json(metric_file), flat)
        except (OSError, json.JSONDecodeError):
            continue
        mappings = {
            "clean_accuracy": ["clean_accuracy", "clean_acc", "accuracy", "metrics.clean_acc"],
            "attack_success_rate": ["attack_success_rate", "asr", "test_asr", "metrics.asr"],
            "defense_asr": ["defense_asr", "stage3_final_asr", "metrics.stage3_final_asr"],
            "corrected_asr": ["corrected_asr", "stage3_final_asr", "metrics.stage3_final_asr"],
            "detection_recall": ["detection_recall", "detection_rate", "metrics.detection_recall"],
            "detection_precision": ["detection_precision", "metrics.detection_precision"],
            "detection_f1": ["detection_f1", "metrics.detection_f1"],
            "false_positive_rate": ["false_positive_rate", "fpr", "metrics.false_positive_rate"],
            "correction_rate": ["correction_rate", "metrics.correction_rate"],
        }
        for output_key, candidates in mappings.items():
            if output_key not in row or row[output_key] == "":
                row[output_key] = first_present(flat, candidates)
    return row


def parse_log(log_path):
    row = {}
    if not log_path.exists():
        return row
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        test_match = TEST_EPOCH_RE.search(line)
        if test_match:
            row.update(test_match.groupdict())
            row["corrected_asr"] = row.get("defense_asr", "")
            continue
        detection_match = DETECTION_RE.search(line)
        if detection_match:
            row.update(detection_match.groupdict())
            continue
        clean_match = CLEAN_DEBUG_RE.search(line)
        if clean_match:
            row.update(clean_match.groupdict())
            continue
        poison_match = POISON_DEBUG_RE.search(line)
        if poison_match:
            row.update(poison_match.groupdict())
            continue
        attack_match = ATTACK_DEBUG_RE.search(line)
        if attack_match:
            row.update(attack_match.groupdict())
    return row


def write_csv(rows, fieldnames, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: as_cell(row.get(key, "")) for key in fieldnames})


def summarize_lambda_anchor(result_root, config_root, output):
    rows = []
    for stage3_dir in sorted(path for path in result_root.glob("lambda_anchor_*/stage3") if path.is_dir()):
        config_path = config_root / stage3_dir.parent.name / "stage3_test.json"
        row = {
            "dataset": "CIFAR10",
            "lambda_anchor": stage3_dir.parent.name.replace("lambda_anchor_", "", 1),
            "stage3_result_directory": stage3_dir.as_posix(),
        }
        if config_path.exists():
            config = load_json(config_path)
            row.update(
                {
                    "dataset": config.get("dataset", "CIFAR10"),
                    "lambda_anchor": config.get("lambda_anchor", row["lambda_anchor"]),
                    "poison_rate": config.get("poison_rate", ""),
                    "lambda_stage1_anchor": config.get("lambda_stage1_anchor", ""),
                    "anchor_margin": config.get("anchor_margin", ""),
                    "anchor_scale": config.get("anchor_scale", ""),
                    "anchor_ema_momentum": config.get("anchor_ema_momentum", ""),
                }
            )
        row.update(parse_metric_files(stage3_dir))
        row.update(parse_log(stage3_dir / "experiment.log"))
        rows.append(row)
    write_csv(rows, LAMBDA_ANCHOR_FIELDS, output)
    return rows


def summarize_lambda_stage1(result_root, config_root, output):
    rows = []
    for stage3_dir in sorted(path for path in result_root.glob("lambda_*/stage3") if path.is_dir()):
        config_path = config_root / stage3_dir.parent.name / "stage3_test.json"
        row = {
            "dataset": "CIFAR10",
            "lambda_stage1_anchor": stage3_dir.parent.name.replace("lambda_", "", 1),
            "stage3_result_directory": stage3_dir.as_posix(),
        }
        if config_path.exists():
            config = load_json(config_path)
            row.update(
                {
                    "dataset": config.get("dataset", "CIFAR10"),
                    "lambda_stage1_anchor": config.get("lambda_stage1_anchor", row["lambda_stage1_anchor"]),
                    "anchor_margin": config.get("anchor_margin", ""),
                    "anchor_scale": config.get("anchor_scale", ""),
                }
            )
        row.update(parse_metric_files(stage3_dir))
        row.update(parse_log(stage3_dir / "experiment.log"))
        rows.append(row)
    write_csv(rows, LAMBDA_STAGE1_FIELDS, output)
    return rows


def summarize_stage1_ablation(result_root, config_root, output):
    rows = []
    for stage3_dir in sorted(path for path in result_root.glob("*/stage3") if path.is_dir()):
        config_path = config_root / stage3_dir.parent.name / "stage3_test.json"
        row = {
            "dataset": "CIFAR10",
            "ablation_variant": stage3_dir.parent.name,
            "stage3_result_directory": stage3_dir.as_posix(),
        }
        if config_path.exists():
            config = load_json(config_path)
            row.update(
                {
                    "dataset": config.get("dataset", "CIFAR10"),
                    "ablation_variant": stage3_dir.parent.name,
                    "enable_stage1": config.get("enable_stage1", ""),
                    "lambda_stage1_anchor": config.get("lambda_stage1_anchor", ""),
                    "anchor_margin": config.get("anchor_margin", ""),
                    "anchor_scale": config.get("anchor_scale", ""),
                    "anchor_ema_momentum": config.get("anchor_ema_momentum", ""),
                }
            )
        row.update(parse_metric_files(stage3_dir))
        row.update(parse_log(stage3_dir / "experiment.log"))
        rows.append(row)
    write_csv(rows, STAGE1_ABLATION_FIELDS, output)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize CIFAR10 AVGuard ablation rerun results.")
    parser.add_argument("--result-root", required=True, help="Root directory containing rerun results")
    parser.add_argument("--config-root", required=True, help="Root directory containing rerun configs")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    config_root = Path(args.config_root)

    lambda_anchor_rows = summarize_lambda_anchor(
        result_root / "lambda_anchor_sensitivity",
        config_root / "lambda_anchor_sensitivity",
        result_root / "lambda_anchor_sensitivity" / "summary.csv",
    )
    lambda_stage1_rows = summarize_lambda_stage1(
        result_root / "lambda_stage1_anchor_sensitivity",
        config_root / "lambda_stage1_anchor_sensitivity",
        result_root / "lambda_stage1_anchor_sensitivity" / "summary.csv",
    )
    stage1_ablation_rows = summarize_stage1_ablation(
        result_root / "stage1_ablation",
        config_root / "stage1_ablation",
        result_root / "stage1_ablation" / "summary.csv",
    )

    print(
        "Wrote summaries: "
        f"lambda_anchor_sensitivity={len(lambda_anchor_rows)}, "
        f"lambda_stage1_anchor_sensitivity={len(lambda_stage1_rows)}, "
        f"stage1_ablation={len(stage1_ablation_rows)}"
    )


if __name__ == "__main__":
    main()
