import argparse
import csv
import math
import re
from pathlib import Path

import torch


TEST_EPOCH_RE = re.compile(
    r"=> Test Epoch: (?P<epoch>\d+), .*? clean acc: (?P<clean_acc>\d+\.\d+), "
    r"Top-(?P<top_k>\d+): (?P<topk>\d+\.\d+), ASR: (?P<asr>\d+\.\d+), RAC: (?P<rac>\d+\.\d+), "
    r"test target accuracy: (?P<target_acc>\d+\.\d+), stage3 final accuracy: (?P<stage3_acc>\d+\.\d+), "
    r"stage3 final target accuracy: (?P<stage3_target_acc>\d+\.\d+), stage3 final asr: (?P<stage3_asr>\d+\.\d+), "
    r"anchor loss: (?P<anchor_loss>\d+\.\d+)"
)
DETECTION_RE = re.compile(
    r"=> Stage 3 Detection Summary: recall: (?P<recall>\d+\.\d+), precision: (?P<precision>\d+\.\d+), "
    r"f1: (?P<f1>\d+\.\d+), false positive rate: (?P<fpr>\d+\.\d+), correction rate: (?P<correction_rate>\d+\.\d+)"
)
END_EPOCH_RE = re.compile(
    r"=> End Epoch: (?P<epoch>\d+), early stop epochs: (?P<early_stop>\d+), best epoch: (?P<best_epoch>\d+), "
    r"best clean acc: (?P<best_clean_acc>\d+\.\d+), best Top-(?P<top_k>\d+): (?P<best_topk>\d+\.\d+), "
    r"best ASR: (?P<best_asr>\d+\.\d+), best RAC: (?P<best_rac>\d+\.\d+), anchor loss: (?P<best_anchor_loss>\d+\.\d+)"
)
PARAM_RE = re.compile(r"(lambda|ema)_(\d+(?:\.\d+)?)")


def _as_float(match_dict, key):
    return float(match_dict[key])


def _find_experiment_dirs(root_dir: Path):
    return sorted(path for path in root_dir.iterdir() if path.is_dir() and (path / "experiment.log").exists())


def _extract_params(experiment_name: str):
    params = {"lambda_anchor": "", "anchor_ema_momentum": ""}
    for kind, value in PARAM_RE.findall(experiment_name):
        if kind == "lambda":
            params["lambda_anchor"] = value
        elif kind == "ema":
            params["anchor_ema_momentum"] = value
    return params


def _parse_log(log_path: Path):
    epoch_metrics = []
    pending_detection_index = None
    trainer_best = None

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        test_match = TEST_EPOCH_RE.search(raw_line)
        if test_match:
            data = test_match.groupdict()
            epoch_metrics.append(
                {
                    "epoch": int(data["epoch"]),
                    "clean_acc": _as_float(data, "clean_acc"),
                    "top_k": int(data["top_k"]),
                    "topk": _as_float(data, "topk"),
                    "asr": _as_float(data, "asr"),
                    "rac": _as_float(data, "rac"),
                    "test_target_acc": _as_float(data, "target_acc"),
                    "stage3_final_accuracy": _as_float(data, "stage3_acc"),
                    "stage3_final_target_accuracy": _as_float(data, "stage3_target_acc"),
                    "stage3_final_asr": _as_float(data, "stage3_asr"),
                    "anchor_loss": _as_float(data, "anchor_loss"),
                    "recall": math.nan,
                    "precision": math.nan,
                    "f1": math.nan,
                    "fpr": math.nan,
                    "correction_rate": math.nan,
                }
            )
            pending_detection_index = len(epoch_metrics) - 1
            continue

        detection_match = DETECTION_RE.search(raw_line)
        if detection_match and pending_detection_index is not None:
            data = detection_match.groupdict()
            epoch_metrics[pending_detection_index].update(
                {
                    "recall": _as_float(data, "recall"),
                    "precision": _as_float(data, "precision"),
                    "f1": _as_float(data, "f1"),
                    "fpr": _as_float(data, "fpr"),
                    "correction_rate": _as_float(data, "correction_rate"),
                }
            )
            pending_detection_index = None
            continue

        end_match = END_EPOCH_RE.search(raw_line)
        if end_match:
            trainer_best = {
                "logged_epoch": int(end_match.group("epoch")),
                "trainer_best_epoch": int(end_match.group("best_epoch")),
                "trainer_best_clean_acc": float(end_match.group("best_clean_acc")),
                "trainer_best_topk": float(end_match.group("best_topk")),
                "trainer_best_asr": float(end_match.group("best_asr")),
                "trainer_best_rac": float(end_match.group("best_rac")),
                "trainer_best_anchor_loss": float(end_match.group("best_anchor_loss")),
                "trainer_top_k": int(end_match.group("top_k")),
                "early_stop_epochs": int(end_match.group("early_stop")),
            }

    if not epoch_metrics:
        raise RuntimeError(f"No stage2 test epochs found in {log_path}")

    max_clean_metrics = max(epoch_metrics, key=lambda item: (item["clean_acc"], -item["epoch"]))
    best_epoch_lookup = {item["epoch"]: item for item in epoch_metrics}
    trainer_best_metrics = best_epoch_lookup.get(trainer_best["trainer_best_epoch"]) if trainer_best else None

    return epoch_metrics, trainer_best, trainer_best_metrics, max_clean_metrics


def _fmt(value):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _build_rows(root_dir: Path):
    rows = []
    for experiment_dir in _find_experiment_dirs(root_dir):
        params = _extract_params(experiment_dir.name)
        epoch_metrics, trainer_best, trainer_best_metrics, max_clean_metrics = _parse_log(experiment_dir / "experiment.log")
        checkpoint_metrics = {}
        checkpoint_path = experiment_dir / "best_checkpoint.pth.tar"
        if not checkpoint_path.exists():
            checkpoint_path = experiment_dir / "best_clean_checkpoint.pth.tar"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_best_metrics = checkpoint.get("best_metrics", {}) or {}
            checkpoint_metrics = {
                "best_checkpoint_epoch": int(checkpoint.get("best_epoch", 0)),
                "best_checkpoint_pre_stage3_acc": checkpoint_best_metrics.get(
                    "pre_stage3_acc",
                    checkpoint_best_metrics.get("test_acc", math.nan),
                ),
                "best_checkpoint_clean_acc": checkpoint_best_metrics.get("clean_acc", math.nan),
                "best_checkpoint_score": checkpoint.get(
                    "best_checkpoint_score",
                    checkpoint_best_metrics.get("best_checkpoint_score", math.nan),
                ),
            }
        row = {
            "experiment": experiment_dir.name,
            "result_dir": experiment_dir.as_posix(),
            "lambda_anchor": params["lambda_anchor"],
            "anchor_ema_momentum": params["anchor_ema_momentum"],
            "best_checkpoint_metric": "mean(clean_acc,detection_recall,correction_rate)",
            "best_checkpoint_epoch": checkpoint_metrics.get(
                "best_checkpoint_epoch",
                trainer_best["trainer_best_epoch"] if trainer_best else "",
            ),
            "best_checkpoint_pre_stage3_acc": checkpoint_metrics.get(
                "best_checkpoint_pre_stage3_acc",
                math.nan,
            ),
            "best_checkpoint_clean_acc": checkpoint_metrics.get(
                "best_checkpoint_clean_acc",
                trainer_best["trainer_best_clean_acc"] if trainer_best else math.nan,
            ),
            "best_checkpoint_score": checkpoint_metrics.get("best_checkpoint_score", math.nan),
            "trainer_best_epoch": trainer_best["trainer_best_epoch"] if trainer_best else "",
            "trainer_best_clean_acc": trainer_best["trainer_best_clean_acc"] if trainer_best else "",
            "trainer_best_topk": trainer_best["trainer_best_topk"] if trainer_best else "",
            "trainer_best_asr": trainer_best["trainer_best_asr"] if trainer_best else "",
            "trainer_best_rac": trainer_best["trainer_best_rac"] if trainer_best else "",
            "trainer_best_anchor_loss": trainer_best["trainer_best_anchor_loss"] if trainer_best else "",
            "trainer_best_recall": trainer_best_metrics["recall"] if trainer_best_metrics else math.nan,
            "trainer_best_precision": trainer_best_metrics["precision"] if trainer_best_metrics else math.nan,
            "trainer_best_f1": trainer_best_metrics["f1"] if trainer_best_metrics else math.nan,
            "trainer_best_fpr": trainer_best_metrics["fpr"] if trainer_best_metrics else math.nan,
            "trainer_best_correction_rate": trainer_best_metrics["correction_rate"] if trainer_best_metrics else math.nan,
            "max_clean_epoch": max_clean_metrics["epoch"],
            "max_clean_acc": max_clean_metrics["clean_acc"],
            "max_clean_topk": max_clean_metrics["topk"],
            "max_clean_asr": max_clean_metrics["asr"],
            "max_clean_rac": max_clean_metrics["rac"],
            "max_clean_anchor_loss": max_clean_metrics["anchor_loss"],
            "max_clean_recall": max_clean_metrics["recall"],
            "max_clean_precision": max_clean_metrics["precision"],
            "max_clean_f1": max_clean_metrics["f1"],
            "max_clean_fpr": max_clean_metrics["fpr"],
            "max_clean_correction_rate": max_clean_metrics["correction_rate"],
            "num_test_epochs": len(epoch_metrics),
        }
        rows.append(row)

    def sort_key(item):
        lambda_value = float(item["lambda_anchor"]) if item["lambda_anchor"] else math.inf
        ema_value = float(item["anchor_ema_momentum"]) if item["anchor_ema_momentum"] else math.inf
        return (lambda_value, ema_value, item["experiment"])

    return sorted(rows, key=sort_key)


def _write_csv(rows, csv_path: Path):
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) for key, value in row.items()})


def _write_md(rows, md_path: Path, title: str):
    lines = [
        f"# {title}",
        "",
        "Best checkpoint selection in the current Stage 2 trainer uses `mean(clean_acc, detection_recall, correction_rate)`.",
        "",
    ]
    if not rows:
        lines.append("No experiment directories with `experiment.log` were found.")
    else:
        headers = [
            "experiment",
            "lambda_anchor",
            "anchor_ema_momentum",
            "best_checkpoint_epoch",
            "best_checkpoint_pre_stage3_acc",
            "best_checkpoint_clean_acc",
            "best_checkpoint_score",
            "trainer_best_epoch",
            "trainer_best_clean_acc",
            "max_clean_epoch",
            "max_clean_acc",
            "trainer_best_asr",
            "trainer_best_rac",
            "trainer_best_recall",
            "trainer_best_precision",
            "trainer_best_f1",
            "trainer_best_fpr",
            "trainer_best_correction_rate",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_fmt(row.get(header, "")) for header in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize AVGuard Stage 2 anchor scan experiments.")
    parser.add_argument("--root_dir", required=True, help="Directory whose direct child experiment folders contain experiment.log")
    parser.add_argument("--output_prefix", required=True, help="Output file prefix without extension")
    parser.add_argument("--title", default="Anchor Scan Summary", help="Markdown title")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    rows = _build_rows(root_dir)
    output_prefix = Path(args.output_prefix)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    _write_csv(rows, csv_path)
    _write_md(rows, md_path, args.title)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
