import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# 1. Data
# =========================
data = [
    ["CIFAR10", 0.25, 0.7051, 0.0588, 0.9864, 0.7135],
    ["CIFAR10", 0.50, 0.7267, 0.0513, 0.9942, 0.7230],
    ["CIFAR10", 0.75, 0.7313, 0.0493, 0.9947, 0.7289],

    ["NUSWIDE", 0.25, 0.7672, 0.1053, 0.8817, 0.9375],
    ["NUSWIDE", 0.50, 0.7692, 0.1019, 0.8865, 0.9394],
    ["NUSWIDE", 0.75, 0.7689, 0.1646, 0.8049, 0.7990],

    ["IEEE-CIS-Fraud", 0.25, 0.8267, 0.0968, 0.9098, 0.9864],
    ["IEEE-CIS-Fraud", 0.50, 0.8196, 0.0929, 0.9170, 0.9890],
    ["IEEE-CIS-Fraud", 0.75, 0.8275, 0.1038, 0.9026, 0.9284],

    ["PHISHING", 0.25, 0.9277, 0.1499, 0.8723, 0.9834],
    ["PHISHING", 0.50, 0.9277, 0.1511, 0.8657, 0.9768],
    ["PHISHING", 0.75, 0.9279, 0.1476, 0.8721, 0.9832],
]

df = pd.DataFrame(
    data,
    columns=[
        "dataset",
        "lambda_s2",
        "clean_accuracy",
        "attack_success_rate",
        "detection_recall",
        "correction_rate",
    ],
)

# =========================
# 2. Plot settings
# =========================
# DataFrame column names (must match df columns above)
metrics = [
    "clean_accuracy",
    "attack_success_rate",
    "detection_recall",
    "correction_rate",
]

metric_labels = [
    "Defense Acc",
    "Defense ASR",
    "Recall",
    "CR",
]

# Similar color palette to your reference figure
colors = [
    "#E68483",  # Clean Acc.
    "#F2B38F",  # ASR
    "#7AA6D1",  # Recall
    "#8FC9A3",  # Correction Rate
]

datasets = ["CIFAR10", "NUSWIDE", "IEEE-CIS-Fraud", "PHISHING"]

# =========================
# 3. Draw one figure with 4 subplots
# =========================
fig, axes = plt.subplots(
    1,
    len(datasets),
    figsize=(18, 3.8),
    sharey=True
)

bar_width = 0.18

for idx, dataset in enumerate(datasets):
    ax = axes[idx]

    sub = df[df["dataset"] == dataset].sort_values("lambda_s2")
    x = np.arange(len(sub))

    for j, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        bars = ax.bar(
            x + (j - 1.5) * bar_width,
            sub[metric].values,
            width=bar_width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.7,
        )

        ax.bar_label(
            bars,
            labels=[f"{v:.3f}" for v in sub[metric].values],
            padding=2,
            fontsize=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(v) for v in sub["lambda_s2"].tolist()],
        fontsize=9
    )

    ax.set_xlabel(r"stage II anchor-regularization coefficient $\lambda_{\mathrm{s2}}$", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    ax.set_title(
        f"({chr(97 + idx)}) {dataset}",
        y=-0.32,
        fontsize=12,
        # fontweight="bold"
    )

    if idx == 0:
        ax.set_ylabel("Metrics", fontsize=10)

    ax.legend(
        fontsize=7,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.18),
        frameon=True
    )

plt.tight_layout()

# =========================
# 4. Save figure
# =========================
plt.savefig(
    "./draw/stage2_lambda_anchor.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()