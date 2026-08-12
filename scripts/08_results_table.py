import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# ── Data ──────────────────────────────────────────────────────────────
results = {
    'Model': [
        'U-Net',
        'SegFormer-B2',
        'SegFormer-B2',
        'SegFormer-B2 ONNX',
        'SegFormer-B2 INT8',
        'SegFormer-B0',
        'SegFormer-B0 ONNX',
        'SegFormer-B2',
        'SegFormer-B0',
    ],
    'Hardware': [
        'M2 CPU',
        'M2 CPU',
        'M2 MPS',
        'M2 CPU',
        'M2 CPU',
        'M2 MPS',
        'M2 CPU',
        'T4 GPU',
        'T4 GPU',
    ],
    'FPS': [
        1.3,
        1.3,
        7.3,
        2.2,
        1.0,
        21.1,
        12.6,
        19.7,
        79.4,
    ],
    'Latency (ms)': [
        769,
        769,
        137,
        450,
        1000,
        47,
        79,
        50,
        12,
    ],
    'mIoU': [
        0.069,
        0.350,
        0.350,
        0.350,
        0.346,
        0.287,
        0.287,
        0.350,
        0.287,
    ],
    'Size (MB)': [
        90,
        105,
        105,
        105,
        29,
        15,
        15,
        105,
        15,
    ],
    'Real-time': [
        '❌',
        '❌',
        '❌',
        '❌',
        '❌',
        '✅',
        '❌',
        '✅',
        '✅',
    ]
}

df = pd.DataFrame(results)

# ── Save CSV ──────────────────────────────────────────────────────────
os.makedirs(os.path.expanduser("~/egoseg_rt/results/"), exist_ok=True)
csv_path = os.path.expanduser("~/egoseg_rt/results/benchmark_table.csv")
df.to_csv(csv_path, index=False)
print(f"CSV saved to {csv_path}")

# ── Print table ───────────────────────────────────────────────────────
print("\n" + "="*75)
print("EgoSeg-RT — Complete Benchmark Table")
print("="*75)
print(df.to_string(index=False))
print("="*75)

# ── Plot FPS comparison ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('EgoSeg-RT — Deployment Benchmark Results',
             fontsize=14, fontweight='bold')

# labels for plot
labels = [
    f"{r['Model']}\n{r['Hardware']}"
    for _, r in df.iterrows()
]

colors = [
    '#e74c3c' if fps < 20 else '#2ecc71'
    for fps in df['FPS']
]

# FPS bar chart
axes[0].barh(labels, df['FPS'], color=colors)
axes[0].axvline(x=20, color='black', linestyle='--',
                linewidth=1.5, label='Real-time (20 FPS)')
axes[0].set_xlabel('FPS (higher is better)')
axes[0].set_title('Inference Speed')
axes[0].legend()

# mIoU bar chart
axes[1].barh(labels, df['mIoU'], color='#3498db')
axes[1].set_xlabel('mIoU (higher is better)')
axes[1].set_title('Segmentation Accuracy')
axes[1].set_xlim(0, 0.5)

plt.tight_layout()

plot_path = os.path.expanduser("~/egoseg_rt/results/benchmark_plot.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {plot_path}")
plt.show()