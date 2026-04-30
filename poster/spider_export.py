import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data ──────────────────────────────────────────────────────────────────────
tasks = ["BoolQ", "PIQA", "Soc. IQA", "HellaSwag",
         "WinoGrande", "ARC-E", "ARC-C", "OBQA"]
N = len(tasks)

lora_full  = [74.0, 84.0, 79.0, 73.5, 83.0, 83.0, 60.3, 82.5]
lora_attn  = [78.0, 83.0, 80.0, 78.0, 84.0, 84.0, 62.0, 88.0]
lora_mlp   = [75.0, 84.0, 79.0, 76.0, 83.0, 85.0, 63.0, 87.0]

dora_full  = [76.0, 85.2, 79.0, 79.5, 82.5, 82.5, 64.0, 84.9]
dora_attn  = [72.0, 82.0, 79.0, 77.0, 83.0, 83.0, 60.0, 88.7]
dora_mlp   = [76.0, 84.0, 79.0, 77.0, 83.0, 85.0, 63.0, 86.6]

VMIN, VMAX = 58.0, 92.0

def normalize(vals):
    return [(v - VMIN) / (VMAX - VMIN) for v in vals]

angles = [np.pi/2 - 2*np.pi*i/N for i in range(N)]
angles_closed = angles + [angles[0]]

def to_xy(norms, angs):
    xs = [r * np.cos(a) for r, a in zip(norms, angs)]
    ys = [r * np.sin(a) for r, a in zip(norms, angs)]
    return xs + [xs[0]], ys + [ys[0]]

# ── Hex colors (Shashwat's codes) ─────────────────────────────────────────────
# index 0 = Full (biggest polygon), 1 = Attn, 2 = MLP (smallest)
ORANGE = ["#D32F2F", "#FB8C00", "#FFEB3B"]   # red, orange, yellow
BLUE   = ["#1A237E", "#1976D2", "#81D4FA"]   # navy, crimson-blue, sky blue

GRID_COLOR  = "#DDDDDD"
RING_VALUES = [66, 74, 82, 90]
RING_RADII  = [(v - VMIN) / (VMAX - VMIN) for v in RING_VALUES]
LABEL_PAD   = 0.26
LEGEND_LABELS = ["Full scope", "Attention only", "MLP only"]

# ── Draw ───────────────────────────────────────────────────────────────────────
def draw_spider(ax, title, datasets, colors, task_labels):
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    # Circular grid
    for r in RING_RADII:
        ax.add_patch(plt.Circle((0, 0), r, color=GRID_COLOR,
                                fill=False, linewidth=1.0, zorder=1))
    for a in angles:
        ax.plot([0, np.cos(a)], [0, np.sin(a)],
                color=GRID_COLOR, linewidth=1.0, zorder=1)

    # Ring % labels
    for r, val in zip(RING_RADII, RING_VALUES):
        ax.text(r * np.cos(angles[0]) + 0.03,
                r * np.sin(angles[0]),
                f"{val}%", fontsize=8, color="#AAAAAA",
                va="center", ha="left", zorder=5)

    # ── Polygons: draw biggest first so smallest sits on top ──────────────────
    # Each polygon uses colors[idx] directly — no hardcoding
    for z, idx in enumerate([0, 1, 2]):          # Full → Attn → MLP
        col   = colors[idx]                       # <-- directly from passed-in list
        vals  = datasets[idx]
        norms = normalize(vals)
        xs, ys = to_xy(norms, angles_closed)

        # Solid fill with the actual hex color inside the shape
        ax.fill(xs, ys,
                facecolor=col, edgecolor="none",
                alpha=0.65, zorder=2 + z)

        # Crisp border line in same color (fully opaque)
        ax.plot(xs, ys, color=col, linewidth=2.5, zorder=10 + z)

        # White vertex dots
        ax.scatter(xs[:-1], ys[:-1], s=80,
                   color="white", edgecolors=col,
                   linewidth=2.0, zorder=16 + z)

    # Task labels
    LR = 1.0 + LABEL_PAD
    for label, ang in zip(task_labels, angles):
        x, y = LR * np.cos(ang), LR * np.sin(ang)
        if   ang > np.pi * 5/8:   ha, va = "right",  "bottom"
        elif ang > np.pi * 3/8:   ha, va = "center", "bottom"
        elif ang > np.pi * 1/8:   ha, va = "left",   "bottom"
        elif ang > -np.pi * 1/8:  ha, va = "left",   "center"
        elif ang > -np.pi * 3/8:  ha, va = "left",   "top"
        elif ang > -np.pi * 5/8:  ha, va = "center", "top"
        elif ang > -np.pi * 7/8:  ha, va = "right",  "top"
        else:                      ha, va = "right",  "center"
        ax.text(x, y, label, fontsize=11, fontweight="bold",
                ha=ha, va=va, color="#222222", zorder=20)

    # Title
    ax.text(0, 1.0 + LABEL_PAD + 0.20, title,
            fontsize=14, fontweight="bold",
            ha="center", va="bottom", color=colors[0], zorder=20)

    # Legend
    handles = [mpatches.Patch(facecolor=colors[i], edgecolor=colors[i],
                               alpha=0.8, linewidth=1.5,
                               label=LEGEND_LABELS[i])
               for i in range(3)]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.14),
              ncol=1, fontsize=10, frameon=False,
              handlelength=1.6, handleheight=1.2)

    ax.set_xlim(-1.60, 1.60)
    ax.set_ylim(-1.55, 1.62)


# ── Export ─────────────────────────────────────────────────────────────────────
for fname, title, datasets, colors in [
    ("lora_spider.png", "LoRA — Per-Task Accuracy by Scope",
     [lora_full, lora_attn, lora_mlp], ORANGE),
    ("dora_spider.png", "DoRA — Per-Task Accuracy by Scope",
     [dora_full, dora_attn, dora_mlp], BLUE),
]:
    fig, ax = plt.subplots(figsize=(8.5, 9.5), facecolor="white")
    draw_spider(ax, title, datasets, colors, tasks)
    plt.tight_layout(pad=0.4)
    fig.savefig(fname, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved {fname}")
