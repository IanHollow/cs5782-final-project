"""
DoRA vs LoRA Analysis: Full / Attention-Only / MLP-Only
CS 5782 Final Project — Spring 2026

Usage:
    python analyze_dora_lora.py                          # uses placeholder data
    python analyze_dora_lora.py --results path/to/runs   # reads real metrics.json files

Outputs (written to ./analysis_output/):
    summary_table.csv        — all 6 runs × 9 metrics
    dora_gains_table.csv     — DoRA − LoRA delta per scope and task
    fig1_macro_grouped.png   — grouped bar chart: macro average by method × scope
    fig2_per_task_heatmap.png— heatmap of accuracy across all 6 conditions
    fig3_dora_gains.png      — horizontal bar chart of DoRA gains per scope
    fig4_spider.png          — radar/spider chart comparing the 6 conditions
    fig5_per_task_bars.png   — per-task side-by-side bars for every task
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Easy",
    "ARC-Challenge",
    "openbookqa",
]

TASK_LABELS = {
    "boolq": "BoolQ",
    "piqa": "PIQA",
    "social_i_qa": "Social IQA",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
    "ARC-Easy": "ARC-Easy",
    "ARC-Challenge": "ARC-Challenge",
    "openbookqa": "OpenBookQA",
}

SCOPES = ["full", "attention_only", "mlp_only"]
SCOPE_LABELS = {
    "full": "Full\n(Attn + MLP)",
    "attention_only": "Attention\nOnly",
    "mlp_only": "MLP\nOnly",
}
METHODS = ["lora", "dora"]

# Style
COLORS = {
    "lora": "#4C72B0",
    "dora": "#DD8452",
}
SCOPE_COLORS = {
    "full": "#2d6a4f",
    "attention_only": "#1d3557",
    "mlp_only": "#9d0208",
}

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# Placeholder data — replace with your real metrics.json files
# ─────────────────────────────────────────────────────────────────────────────

PLACEHOLDER_DATA: list[dict] = [
    # Llama-2-7B, colab_a100_40gb — real results from benchmark_summary sheet
    {
        "run_name": "lora_full",
        "method": "lora",
        "scope": "full",
        "model_name": "llama2_7b",
        "boolq": 0.8186544343,
        "piqa": 0.7867247008,
        "social_i_qa": 0.7548618219,
        "hellaswag": 0.7639912743,
        "winogrande": 0.7521704815,
        "ARC-Easy": 0.8614035088,
        "ARC-Challenge": 0.6889632107,
        "openbookqa": 0.766,
        "macro_average": 0.7740961743,
    },
    {
        "run_name": "dora_full",
        "method": "dora",
        "scope": "full",
        "model_name": "llama2_7b",
        "boolq": 0.8311926606,
        "piqa": 0.7872687704,
        "social_i_qa": 0.7548618219,
        "hellaswag": 0.8239394543,
        "winogrande": 0.7513812155,
        "ARC-Easy": 0.8719298246,
        "ARC-Challenge": 0.7257525084,
        "openbookqa": 0.79,
        "macro_average": 0.7920407819,
    },
    {
        "run_name": "lora_attention_only",
        "method": "lora",
        "scope": "attention_only",
        "model_name": "llama2_7b",
        "boolq": 0.829969419,
        "piqa": 0.7981501632,
        "social_i_qa": 0.775332651,
        "hellaswag": 0.8025293766,
        "winogrande": 0.7411207577,
        "ARC-Easy": 0.8438596491,
        "ARC-Challenge": 0.7658862876,
        "openbookqa": 0.81,
        "macro_average": 0.795856038,
    },
    {
        "run_name": "dora_attention_only",
        "method": "dora",
        "scope": "attention_only",
        "model_name": "llama2_7b",
        "boolq": 0.8229357798,
        "piqa": 0.78781284,
        "social_i_qa": 0.7599795292,
        "hellaswag": 0.7827126071,
        "winogrande": 0.72691397,
        "ARC-Easy": 0.8456140351,
        "ARC-Challenge": 0.7391304348,
        "openbookqa": 0.782,
        "macro_average": 0.7808873995,
    },
    {
        "run_name": "lora_mlp_only",
        "method": "lora",
        "scope": "mlp_only",
        "model_name": "llama2_7b",
        "boolq": 0.8330275229,
        "piqa": 0.8112078346,
        "social_i_qa": 0.7784032753,
        "hellaswag": 0.8073093009,
        "winogrande": 0.7442778216,
        "ARC-Easy": 0.8543859649,
        "ARC-Challenge": 0.6989966555,
        "openbookqa": 0.796,
        "macro_average": 0.790451047,
    },
    {
        "run_name": "dora_mlp_only",
        "method": "dora",
        "scope": "mlp_only",
        "model_name": "llama2_7b",
        "boolq": 0.8415902141,
        "piqa": 0.7959738847,
        "social_i_qa": 0.7604912999,
        "hellaswag": 0.7836088429,
        "winogrande": 0.7569060773,
        "ARC-Easy": 0.8701754386,
        "ARC-Challenge": 0.7491638796,
        "openbookqa": 0.778,
        "macro_average": 0.7919887046,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: Path | None) -> pd.DataFrame:
    """Load metrics from results/runs/*/metrics.json, or fall back to placeholder."""
    rows: list[dict] = []

    if results_dir is not None and results_dir.exists():
        for metrics_path in sorted(results_dir.glob("*/metrics.json")):
            try:
                rows.append(json.loads(metrics_path.read_text()))
            except Exception as exc:
                print(f"  [warn] Could not read {metrics_path}: {exc}", file=sys.stderr)

    if not rows:
        print("  [info] No metrics.json files found — using placeholder data.", file=sys.stderr)
        print("         Replace PLACEHOLDER_DATA or pass --results <dir> to use real results.\n",
              file=sys.stderr)
        rows = PLACEHOLDER_DATA

    df = pd.DataFrame(rows)

    # Ensure required columns exist
    for col in ["method", "scope", "macro_average"] + TASKS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in metrics data.")

    # Normalise scope/method to lowercase
    df["method"] = df["method"].str.lower()
    df["scope"] = df["scope"].str.lower()

    return df


def compute_gains(df: pd.DataFrame) -> pd.DataFrame:
    """DoRA − LoRA delta for each scope and task."""
    records = []
    for scope in SCOPES:
        lora_row = df[(df["method"] == "lora") & (df["scope"] == scope)]
        dora_row = df[(df["method"] == "dora") & (df["scope"] == scope)]
        if lora_row.empty or dora_row.empty:
            continue
        row: dict = {"scope": scope}
        for task in TASKS:
            row[task] = float(dora_row[task].values[0]) - float(lora_row[task].values[0])
        row["macro_average"] = (
            float(dora_row["macro_average"].values[0])
            - float(lora_row["macro_average"].values[0])
        )
        records.append(row)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig1_macro_grouped(df: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart: macro-average accuracy by method × scope."""
    scopes_present = [s for s in SCOPES if s in df["scope"].values]
    x = np.arange(len(scopes_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, method in enumerate(METHODS):
        vals = []
        for scope in scopes_present:
            row = df[(df["method"] == method) & (df["scope"] == scope)]
            vals.append(float(row["macro_average"].values[0]) if not row.empty else 0.0)
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=method.upper(), color=COLORS[method], alpha=0.88,
                      edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([SCOPE_LABELS[s] for s in scopes_present], fontsize=10)
    ax.set_ylabel("Macro-Average Accuracy", fontsize=11)
    ax.set_title("DoRA vs LoRA — Macro-Average Accuracy by Scope", fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.0, df["macro_average"].max() * 1.12))
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig2_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Heatmap of per-task accuracy for all 6 conditions."""
    run_order = [
        (m, s)
        for s in SCOPES
        for m in METHODS
        if not df[(df["method"] == m) & (df["scope"] == s)].empty
    ]
    row_labels = [f"{m.upper()}\n{SCOPE_LABELS[s]}" for m, s in run_order]
    col_labels = [TASK_LABELS[t] for t in TASKS]

    data = np.array([
        [float(df[(df["method"] == m) & (df["scope"] == s)][t].values[0])
         for t in TASKS]
        for m, s in run_order
    ])

    fig, ax = plt.subplots(figsize=(13, len(run_order) * 0.9 + 1.5))
    im = ax.imshow(data, cmap="YlGn", vmin=data.min() - 0.02, vmax=min(1.0, data.max() + 0.02),
                   aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Per-Task Accuracy Heatmap — All 6 Conditions", fontsize=13, fontweight="bold",
                 pad=12)

    for i in range(len(run_order)):
        for j in range(len(TASKS)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if data[i, j] < 0.85 else "white")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Accuracy")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig3_dora_gains(gains: pd.DataFrame, out: Path) -> None:
    """Horizontal bar chart — DoRA gain over LoRA per scope, with per-task breakdown."""
    scopes_present = [s for s in SCOPES if s in gains["scope"].values]
    n_scopes = len(scopes_present)
    fig, axes = plt.subplots(1, n_scopes, figsize=(5 * n_scopes, 5), sharey=True)
    if n_scopes == 1:
        axes = [axes]

    all_cols = TASKS + ["macro_average"]
    labels = [TASK_LABELS[t] for t in TASKS] + ["MACRO AVG"]

    for ax, scope in zip(axes, scopes_present):
        row = gains[gains["scope"] == scope].iloc[0]
        vals = [float(row[c]) for c in all_cols]
        colors = [
            (SCOPE_COLORS[scope] if v >= 0 else "#c1121f")
            for v in vals
        ]
        # Make macro avg bar a bit darker by adjusting its color directly; skip alpha list
        bars = ax.barh(range(len(labels)), vals, color=colors,
                       edgecolor="white", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.8)

        for bar, val in zip(bars, vals):
            xpos = val + 0.001 if val >= 0 else val - 0.001
            ha = "left" if val >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=8)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{SCOPE_LABELS[scope].replace(chr(10), ' ')}", fontsize=11,
                     fontweight="bold", color=SCOPE_COLORS[scope])
        ax.set_xlabel("Δ Accuracy (DoRA − LoRA)", fontsize=9)

    fig.suptitle("DoRA Gains Over LoRA — Per Scope & Task", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig4_spider(df: pd.DataFrame, out: Path) -> None:
    """Radar chart comparing all 6 conditions across 8 tasks."""
    angles = np.linspace(0, 2 * np.pi, len(TASKS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    linestyles = {"lora": "--", "dora": "-"}
    scope_marker = {"full": "o", "attention_only": "s", "mlp_only": "^"}

    for scope in SCOPES:
        for method in METHODS:
            row = df[(df["method"] == method) & (df["scope"] == scope)]
            if row.empty:
                continue
            vals = [float(row[t].values[0]) for t in TASKS]
            vals += vals[:1]
            color = COLORS[method]
            ax.plot(angles, vals,
                    linestyle=linestyles[method],
                    color=color,
                    linewidth=1.8 if scope == "full" else 1.2,
                    alpha=0.85,
                    marker=scope_marker[scope],
                    markersize=5,
                    label=f"{method.upper()} / {scope.replace('_', ' ')}")
            ax.fill(angles, vals, color=color, alpha=0.04)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9"], fontsize=7)
    ax.set_title("Radar: Per-Task Accuracy — All 6 Conditions", fontsize=13,
                 fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def fig5_per_task_bars(df: pd.DataFrame, out: Path) -> None:
    """8-panel figure: one bar cluster per task, showing all 6 conditions."""
    n_tasks = len(TASKS)
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes_flat = axes.flatten()

    run_order = [(m, s) for s in SCOPES for m in METHODS]
    run_labels = [f"{m.upper()}\n{s.replace('_', ' ')}" for m, s in run_order]
    run_colors = [COLORS[m] for m, _ in run_order]
    scope_alphas = {"full": 1.0, "attention_only": 0.7, "mlp_only": 0.45}

    for i, task in enumerate(TASKS):
        ax = axes_flat[i]
        vals = []
        for method, scope in run_order:
            row = df[(df["method"] == method) & (df["scope"] == scope)]
            vals.append(float(row[task].values[0]) if not row.empty else 0.0)

        x = np.arange(len(run_order))
        bar_colors = [COLORS[m] for m, _ in run_order]
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=0.6)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_title(TASK_LABELS[task], fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, fontsize=6, rotation=45, ha="right")
        lo = max(0, min(vals) - 0.05)
        ax.set_ylim(lo, min(1.0, max(vals) + 0.07))
        ax.set_ylabel("Accuracy", fontsize=8)

    # Hide unused panels
    for j in range(n_tasks, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Add legend
    patches = [
        mpatches.Patch(color=COLORS["lora"], label="LoRA"),
        mpatches.Patch(color=COLORS["dora"], label="DoRA"),
    ]
    fig.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)
    fig.suptitle("Per-Task Accuracy: All 6 Conditions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, gains: pd.DataFrame) -> None:
    print("\n" + "═" * 70)
    print("  SUMMARY TABLE — Macro-Average Accuracy")
    print("═" * 70)
    print(f"  {'Scope':<20} {'LoRA':>10} {'DoRA':>10} {'Δ (DoRA−LoRA)':>14}")
    print("  " + "─" * 56)
    for scope in SCOPES:
        lora_row = df[(df["method"] == "lora") & (df["scope"] == scope)]
        dora_row = df[(df["method"] == "dora") & (df["scope"] == scope)]
        if lora_row.empty or dora_row.empty:
            continue
        lora_acc = float(lora_row["macro_average"].values[0])
        dora_acc = float(dora_row["macro_average"].values[0])
        delta = dora_acc - lora_acc
        sign = "+" if delta >= 0 else ""
        print(f"  {scope:<20} {lora_acc:>10.4f} {dora_acc:>10.4f} {sign+f'{delta:.4f}':>14}")
    print("═" * 70)

    if not gains.empty:
        print("\n  ABLATION INSIGHT — Where does DoRA help most?")
        print("  " + "─" * 56)
        best_scope = gains.loc[gains["macro_average"].idxmax(), "scope"]
        best_gain = gains["macro_average"].max()
        print(f"  Largest macro gain: {best_scope} ({best_gain:+.4f})")
        print("\n  Per-task DoRA gains by scope:")
        for task in TASKS:
            task_gains = {
                row["scope"]: row[task]
                for _, row in gains.iterrows()
            }
            best = max(task_gains, key=task_gains.get)
            vals_str = "  ".join(f"{s[:4]}: {v:+.3f}" for s, v in task_gains.items())
            print(f"    {TASK_LABELS[task]:<16} {vals_str}   ← best: {best}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse DoRA vs LoRA results.")
    parser.add_argument(
        "--results", type=Path, default=None,
        help="Path to results/runs/ directory containing */metrics.json files."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("analysis_output"),
        help="Directory to write figures and CSV tables (default: analysis_output/)."
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading results...")

    df = load_results(args.results)
    gains = compute_gains(df)

    # ── CSV exports ──────────────────────────────────────────────────────────
    summary_cols = ["method", "scope", "macro_average"] + TASKS
    df[summary_cols].to_csv(out_dir / "summary_table.csv", index=False)
    print(f"  Saved: summary_table.csv")

    if not gains.empty:
        gains.to_csv(out_dir / "dora_gains_table.csv", index=False)
        print(f"  Saved: dora_gains_table.csv")

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig1_macro_grouped(df, out_dir / "fig1_macro_grouped.png")
    fig2_heatmap(df, out_dir / "fig2_per_task_heatmap.png")
    if not gains.empty:
        fig3_dora_gains(gains, out_dir / "fig3_dora_gains.png")
    fig4_spider(df, out_dir / "fig4_spider.png")
    fig5_per_task_bars(df, out_dir / "fig5_per_task_bars.png")

    # ── Console summary ──────────────────────────────────────────────────────
    print_summary(df, gains)
    print(f"All outputs written to:  {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
