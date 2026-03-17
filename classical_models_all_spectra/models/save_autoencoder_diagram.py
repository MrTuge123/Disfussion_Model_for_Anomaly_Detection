import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

os.makedirs('../figures', exist_ok=True)

# ── Layer definitions ────────────────────────────────────────────────────────
layers = [
    {'label': 'Input\n535',        'color': '#2c3e50', 'text': 'white'},
    {'label': 'Encoder\n512',      'color': '#2980b9', 'text': 'white'},
    {'label': 'Encoder\n256',      'color': '#2980b9', 'text': 'white'},
    {'label': 'Encoder\n128',      'color': '#2980b9', 'text': 'white'},
    {'label': 'Latent\n64',        'color': '#e74c3c', 'text': 'white'},
    {'label': 'Decoder\n128',      'color': '#27ae60', 'text': 'white'},
    {'label': 'Decoder\n256',      'color': '#27ae60', 'text': 'white'},
    {'label': 'Decoder\n512',      'color': '#27ae60', 'text': 'white'},
    {'label': 'Output\n535',       'color': '#2c3e50', 'text': 'white'},
]

dims      = [535, 512, 256, 128, 64, 128, 256, 512, 535]
n_layers  = len(layers)
max_dim   = max(dims)

# Box geometry
box_w   = 1.1
x_gap   = 0.55
total_w = n_layers * box_w + (n_layers - 1) * x_gap

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(-0.5, total_w + 0.5)
ax.set_ylim(-1.2, 5.5)
ax.axis('off')

box_positions = []

for i, (layer, dim) in enumerate(zip(layers, dims)):
    # Scale box height proportional to dimension
    h     = 0.8 + 3.8 * (dim / max_dim)
    x     = i * (box_w + x_gap)
    y     = (4.6 - h) / 2          # centre vertically

    box = mpatches.FancyBboxPatch(
        (x, y), box_w, h,
        boxstyle='round,pad=0.08',
        facecolor=layer['color'],
        edgecolor='white',
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(box)

    # Layer label inside box
    ax.text(x + box_w / 2, y + h / 2, layer['label'],
            ha='center', va='center',
            fontsize=9.5, fontweight='bold',
            color=layer['text'], zorder=4)

    box_positions.append((x, y, h))

# ── Arrows between boxes ─────────────────────────────────────────────────────
for i in range(n_layers - 1):
    x0, y0, h0 = box_positions[i]
    x1, y1, h1 = box_positions[i + 1]

    mid_y = (4.6 - min(h0, h1)) / 2 + min(h0, h1) / 2

    ax.annotate('',
        xy     =(x1,          mid_y),
        xytext =(x0 + box_w,  mid_y),
        arrowprops=dict(arrowstyle='->', color='#555555',
                        lw=1.8, mutation_scale=16),
        zorder=2
    )

    # Activation label on arrow (ReLU between hidden layers, none on latent→decoder)
    is_latent_out = (i == 3)   # latent has no activation
    is_output_in  = (i == 7)   # output layer has no activation
    if not is_latent_out and not is_output_in:
        ax.text(x0 + box_w + x_gap / 2, mid_y + 0.22,
                'ReLU', ha='center', va='bottom',
                fontsize=7, color='#888888', style='italic')

# ── Section labels (Encoder / Bottleneck / Decoder) ──────────────────────────
enc_x_start = box_positions[1][0]
enc_x_end   = box_positions[3][0] + box_w
ax.annotate('', xy=(enc_x_end, -0.55), xytext=(enc_x_start, -0.55),
            arrowprops=dict(arrowstyle='<->', color='#2980b9', lw=1.5))
ax.text((enc_x_start + enc_x_end) / 2, -0.85,
        'Encoder', ha='center', fontsize=10, color='#2980b9', fontweight='bold')

bot_x = box_positions[4][0]
ax.text(bot_x + box_w / 2, -0.85,
        'Bottleneck', ha='center', fontsize=10, color='#e74c3c', fontweight='bold')

dec_x_start = box_positions[5][0]
dec_x_end   = box_positions[7][0] + box_w
ax.annotate('', xy=(dec_x_end, -0.55), xytext=(dec_x_start, -0.55),
            arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.5))
ax.text((dec_x_start + dec_x_end) / 2, -0.85,
        'Decoder', ha='center', fontsize=10, color='#27ae60', fontweight='bold')

# ── MSE loss annotation ───────────────────────────────────────────────────────
inp_x = box_positions[0][0] + box_w / 2
out_x = box_positions[-1][0] + box_w / 2
ax.annotate('',
    xy     =(out_x, 4.95),
    xytext =(inp_x, 4.95),
    arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=1.5,
                    connectionstyle='arc3,rad=-0.25'),
    zorder=1
)
ax.text((inp_x + out_x) / 2, 5.25,
        'MSE Loss  =  mean( x − x̂ )²',
        ha='center', fontsize=9.5, color='#8e44ad', fontweight='bold')

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title('Autoencoder Architecture — Anomaly Detection on Raman Spectra',
             fontsize=13, fontweight='bold', pad=14)

plt.tight_layout()
plt.savefig('../figures/autoencoder_diagram.png', dpi=150, bbox_inches='tight')
print('Saved to ../figures/autoencoder_diagram.png')
