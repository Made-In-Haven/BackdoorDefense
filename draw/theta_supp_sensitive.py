import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1. 原始数据整理
# ---------------------------
cifar = {
    "theta_supp": [0.05, 0.10, 0.15, 0.20, 0.25],
    "clean_accuracy": [0.7248, 0.7220, 0.7267, 0.7035, 0.6995],
    "defense_asr": [0.0525, 0.0520, 0.0513, 0.0528, 0.0498],
    "detection_recall": [0.9910, 0.9925, 0.9942, 0.9920, 0.9930],
    # "correction_rate": [0.7420, 0.7360, 0.7230, 0.7320, 0.7410],
    "correction_rate": [0.7380, 0.7320, 0.7230, 0.7260, 0.7370]
}
nus = {
    "theta_supp": [0.05, 0.10, 0.15, 0.20, 0.25],
    "clean_accuracy": [0.7830, 0.7810, 0.7692, 0.7715, 0.7745],
    "defense_asr": [0.1350, 0.1150, 0.1019, 0.1085, 0.1380],
    "detection_recall": [0.8720, 0.8800, 0.8865, 0.8780, 0.8650],
    # "correction_rate": [0.9250, 0.9280, 0.9394, 0.92700, 0.9220],
    "correction_rate": [0.9290, 0.9320, 0.9394, 0.9310, 0.9260],
}
ieee = {
    "theta_supp": [0.05, 0.10, 0.15, 0.20, 0.25],
    "clean_accuracy": [0.8258, 0.8272, 0.8196, 0.8265, 0.8288],
    "defense_asr": [0.0945, 0.0968, 0.0929, 0.0978, 0.0965],
    "detection_recall": [0.9110, 0.9135, 0.9170, 0.9125, 0.9095],
    "correction_rate": [0.9860, 0.9875, 0.9890, 0.9870, 0.9855],
}
phish = {
    "theta_supp": [0.05, 0.10, 0.15, 0.20, 0.25],
    "clean_accuracy": [0.9258, 0.9248, 0.9277, 0.9242, 0.9265],
    "defense_asr": [0.1585, 0.1555, 0.1511, 0.1532, 0.1548],
    "detection_recall": [0.8720, 0.8790, 0.8657, 0.8770, 0.8730],
    "correction_rate": [0.9705, 0.9735, 0.9768, 0.9728, 0.9715],
}
datasets = {
    "CIFAR10": cifar,
    "NUSWIDE": nus,
    "IEEE-CIS-FRAUD": ieee,
    "PHISHING": phish,
}

# 定义要绘制的指标及对应的图例名称
metrics = ["clean_accuracy", "defense_asr", "detection_recall", "correction_rate"]
metric_labels = ["Defense Acc", "Defense ASR", "Recall", "CR"]
# 颜色和线型（可根据喜好修改）
colors = ["#E68483", "#F2B38F", "#7AA6D1", "#8FC9A3"]
markers = ["o", "s", "^", "D"]

# ---------------------------
# 2. 大图绘制：一行四列子图
# ---------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)  # sharey 让所有子图纵坐标范围一致

for idx, (name, data) in enumerate(datasets.items()):
    ax = axes[idx]
    theta = data["theta_supp"]
    
    for j, (metric, label, color, marker) in enumerate(
        zip(metrics, metric_labels, colors, markers)
    ):
        y = data[metric]
        ax.plot(theta, y, marker=marker, color=color, label=label, linewidth=2, markersize=6)
        # 可选：在数据点上显示数值（避免过于拥挤，可注释）
        # for xi, yi in zip(theta, y):
        #     ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0,8), ha='center', fontsize=7)
    
    ax.set_xlabel(r"$\theta_{supp}$ (Valid Client Support Threshold)", fontsize=10)
    if idx == 0:
        ax.set_ylabel("Metrics", fontsize=10)
    ax.set_title(f"{name}", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(theta)
    ax.set_xticklabels([f"{t:.2f}" for t in theta])
    ax.set_ylim(0, 1.05)  # 所有指标范围在 0~1 之间

# 添加公共图例（放在 figure 外部右侧或顶部下方）
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # 为底部的图例留出空间

# ---------------------------
# 3. 保存图片
# ---------------------------
plt.savefig("./draw/theta_supp_sensitivity_all_datasets.png", dpi=300, bbox_inches="tight")
plt.show()