import matplotlib.pyplot as plt
import numpy as np

# 数据准备
datasets = ['CIFAR10', 'NUS-WIDE', 'PHISHING', 'IEEE-CIS-Fraud']
asr_with = [0.0513, 0.1019, 0.1511, 0.0929]
asr_without = [0.097, 0.1274, 0.1627, 0.1795]
cr_with = [0.723, 0.9394, 0.9768, 0.989]
cr_without = [0.6697, 0.7691, 0.9712, 0.8782]

x = np.arange(len(datasets))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# ---------- 上图：ASR ----------
bars1 = ax1.bar(x - width/2, asr_with, width, label='with Weighted Voting',
                color='lightcoral', edgecolor='black')
bars2 = ax1.bar(x + width/2, asr_without, width, label='without Weighted Voting',
                color='steelblue', edgecolor='black')

ax1.set_ylabel('ASR', fontsize=12)
# ax1.set_title('ASR and Correction Rate Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=11)
ax1.legend(fontsize=9)                    # 字号调小一号（原10）
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.4f}', ha='center', va='bottom', fontsize=8)  # 字号调小一号（原9）

# ---------- 下图：Correction Rate ----------
bars3 = ax2.bar(x - width/2, cr_with, width, label='with Weighted Voting',
                color="#E68483", edgecolor='black')
bars4 = ax2.bar(x + width/2, cr_without, width, label='without Weighted Voting',
                color="#7AA6D1", edgecolor='black')

ax2.set_ylabel('Correction Rate', fontsize=12)
ax2.set_xlabel('Datasets', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=11)
ax2.legend(fontsize=9)                    # 字号调小一号
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars3 + bars4:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.4f}', ha='center', va='bottom', fontsize=8)  # 字号调小一号

plt.tight_layout()
plt.savefig('comparison_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()