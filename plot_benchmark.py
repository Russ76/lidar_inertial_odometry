#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Labels: Method (Sensor)
labels = [
    'FAST-LIO2\n(AVIA)', 'FASTER-LIO\n(AVIA)', 'Surfel-LIO\n(AVIA)',
    'FAST-LIO2\n(Mid360)', 'FASTER-LIO\n(Mid360)', 'Surfel-LIO\n(Mid360)'
]

# Data
rmse = [0.397, 0.362, 0.365, 0.342, 0.352, 0.342]
fps = [125, 184, 531, 282, 353, 690]

# Colors: FAST-LIO2, FASTER-LIO, Surfel-LIO (repeated for 2 sensors)
colors = ['#4A90D9', '#F5A623', '#7ED321', '#4A90D9', '#F5A623', '#7ED321']

fig, ax1 = plt.subplots(figsize=(5, 4.5))
ax2 = ax1.twinx()

x = np.arange(len(labels))
width = 0.35

# RMSE bars (left y-axis)
bars1 = ax1.bar(x - width/2, rmse, width, color=colors, alpha=0.7, edgecolor='black', label='APE RMSE (m)')
ax1.set_ylabel('APE RMSE (m)', fontsize=10, color='black')
ax1.set_ylim(0, 0.5)
ax1.tick_params(axis='y', labelcolor='black')

# FPS bars (right y-axis)
bars2 = ax2.bar(x + width/2, fps, width, color=colors, hatch='///', edgecolor='black', label='FPS')
ax2.set_ylabel('FPS', fontsize=10, color='black')
ax2.set_ylim(0, 800)
ax2.tick_params(axis='y', labelcolor='black')

# X-axis with rotated labels
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')

# Add vertical separator line between AVIA and Mid360
ax1.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Value labels on bars
for bar, val in zip(bars1, rmse):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
             ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars2, fps):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{val}', 
             ha='center', va='bottom', fontsize=7, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='APE RMSE (m)'),
    Patch(facecolor='gray', hatch='///', edgecolor='black', label='FPS')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('benchmark_comparison.pdf', bbox_inches='tight')
print("Saved: benchmark_comparison.png, benchmark_comparison.pdf")
plt.show()
