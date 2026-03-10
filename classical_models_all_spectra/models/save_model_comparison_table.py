import matplotlib.pyplot as plt
import os

os.makedirs('../figures', exist_ok=True)

rows = [
    ['Autoencoder',                       '0.972', '0.818', '45', '10', '61'],
    ['OC-SVM (nu=0.05, γ=0.001)',         '0.947', '0.727', '40', '15', '61'],
    ['kNN (k=3, euclidean)',              '0.922', '0.455', '25', '30', '61'],
    ['Isolation Forest (n=100, feat=0.25)', '0.754', '0.127',  '7', '48', '61'],
    ['PCA Reconstruction',                '0.795', '0.073',  '4', '51', '61'],
]

col_labels = ['Model', 'AUC', 'Recall', 'TP', 'FN', 'FP']
col_widths  = [0.34, 0.10, 0.10, 0.08, 0.08, 0.08]

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis('off')

table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.0)

# Set column widths
for col_idx, width in enumerate(col_widths):
    for row_idx in range(len(rows) + 1):
        table[row_idx, col_idx].set_width(width)

# Header styling
header_color = '#2c3e50'
for col_idx in range(len(col_labels)):
    cell = table[0, col_idx]
    cell.set_facecolor(header_color)
    cell.set_text_props(color='white', fontweight='bold')

# Row colours — highlight rank 1 and 2, fade out rank 4 and 5
row_colors = ['#d4edda', '#d0e8f5', '#ffffff', '#fff3cd', '#fde8e8']
for row_idx, color in enumerate(row_colors, start=1):
    for col_idx in range(len(col_labels)):
        table[row_idx, col_idx].set_facecolor(color)

# Bold the Recall column values
for row_idx in range(1, len(rows) + 1):
    table[row_idx, 2].set_text_props(fontweight='bold')

ax.set_title('Model Comparison — Anomaly Detection\n'
             '(ranked by Recall, threshold = p95 of normal training scores)',
             fontsize=12, fontweight='bold', pad=16)

plt.tight_layout()
plt.savefig('../figures/model_comparison_table.png', dpi=150, bbox_inches='tight')
print('Saved to ../figures/model_comparison_table.png')
