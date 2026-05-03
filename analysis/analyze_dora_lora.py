"""Create analysis tables and Seaborn figures for the DoRA vs LoRA reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from matplotlib.container import BarContainer

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
    "full": "Full",
    "attention_only": "Attention only",
    "mlp_only": "MLP only",
}
METHODS = ["lora", "dora"]
METHOD_LABELS = {"lora": "LoRA", "dora": "DoRA"}
LORAX_ORANGE = "#FDB945"
LORAX_ORANGE_LIGHT = "#FFF7E3"
DORAEMON_BLUE = "#009AD8"
DORAEMON_BLUE_DARK = "#0063C1"
DARK_TEXT = "#17212F"
GRID_COLOR = "#D7DEE8"
PALETTE = {"LoRA": LORAX_ORANGE, "DoRA": DORAEMON_BLUE}


class ScopeSummaryRecord(TypedDict):
    scope: str
    scope_label: str
    lora_macro: float
    dora_macro: float
    delta: float
    delta_points: float
    task_wins: int
    task_ties: int
    task_losses: int
    best_task: str
    worst_task: str
    mean_task_delta_points: float
    median_task_delta_points: float
    task_delta_std_points: float


OFFICIAL_LLAMA2_7B_REFERENCE = [
    {
        "source": "DoRA paper Table 1",
        "method": "lora",
        "paper_method": "LoRA",
        "rank": 32,
        "params_percent": 0.83,
        "boolq": 0.698,
        "piqa": 0.799,
        "social_i_qa": 0.795,
        "hellaswag": 0.836,
        "winogrande": 0.826,
        "ARC-Easy": 0.798,
        "ARC-Challenge": 0.647,
        "openbookqa": 0.810,
        "macro_average": 0.776,
    },
    {
        "source": "DoRA paper Table 1",
        "method": "dora",
        "paper_method": "DoRA-dagger (Ours)",
        "rank": 16,
        "params_percent": 0.43,
        "boolq": 0.720,
        "piqa": 0.831,
        "social_i_qa": 0.799,
        "hellaswag": 0.891,
        "winogrande": 0.830,
        "ARC-Easy": 0.845,
        "ARC-Challenge": 0.710,
        "openbookqa": 0.812,
        "macro_average": 0.805,
    },
]

CHECKED_IN_SUMMARY_PATH = Path("results/analysis/summary_table.csv")
CHECKED_IN_RANKED_PATH = Path("results/analysis/benchmark_summary_rank.csv")


def _configure_plot_style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 22,
            "axes.labelsize": 17,
            "axes.labelcolor": DARK_TEXT,
            "text.color": DARK_TEXT,
            "xtick.color": "#293548",
            "ytick.color": "#293548",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 1.15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
    )


def _first_float(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].to_numpy()[0])


def _row_for(df: pd.DataFrame, method: str, scope: str) -> pd.DataFrame:
    return df[(df["method"] == method) & (df["scope"] == scope)]


def _condition_label(method: str, scope: str) -> str:
    return f"{METHOD_LABELS[method]} - {SCOPE_LABELS[scope]}"


def _save(fig: plt.Figure, out: Path, *, tight: bool = True, dpi: int | None = None) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def load_results(results_dir: Path | None) -> pd.DataFrame:
    """Load metrics from results/runs/*/metrics.json, or use checked-in values."""
    rows: list[dict[str, Any]] = []

    if results_dir is not None and results_dir.exists():
        for metrics_path in sorted(results_dir.glob("*/metrics.json")):
            try:
                rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  [warn] Could not read {metrics_path}: {exc}", file=sys.stderr)

    if rows:
        df = pd.DataFrame(rows)
    elif CHECKED_IN_SUMMARY_PATH.exists():
        print(
            f"  [info] No metrics.json files found; using {CHECKED_IN_SUMMARY_PATH}.",
            file=sys.stderr,
        )
        df = pd.read_csv(CHECKED_IN_SUMMARY_PATH)
    else:
        msg = (
            "No metrics.json files found and results/analysis/summary_table.csv is missing. "
            "Pass --results or run an experiment first."
        )
        raise FileNotFoundError(msg)

    for column in ["method", "scope", "macro_average", *TASKS]:
        if column not in df.columns:
            msg = f"Missing required column '{column}' in metrics data."
            raise ValueError(msg)

    df["method"] = df["method"].str.lower()
    df["scope"] = df["scope"].str.lower()
    df["method_label"] = df["method"].map(METHOD_LABELS)
    df["scope_label"] = df["scope"].map(SCOPE_LABELS)
    df["condition"] = list(starmap(_condition_label, zip(df["method"], df["scope"], strict=True)))
    return df


def compute_gains(df: pd.DataFrame) -> pd.DataFrame:
    """Compute DoRA minus LoRA deltas for each scope and task."""
    records: list[dict[str, Any]] = []
    for scope in SCOPES:
        lora_row = _row_for(df, "lora", scope)
        dora_row = _row_for(df, "dora", scope)
        if lora_row.empty or dora_row.empty:
            continue
        row: dict[str, Any] = {"scope": scope, "scope_label": SCOPE_LABELS[scope]}
        for task in TASKS:
            row[task] = _first_float(dora_row, task) - _first_float(lora_row, task)
        row["macro_average"] = _first_float(dora_row, "macro_average") - _first_float(
            lora_row, "macro_average"
        )
        records.append(row)
    return pd.DataFrame(records)


def to_long_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide run metrics to task-level long form."""
    long = df.melt(
        id_vars=["method", "scope", "method_label", "scope_label", "condition", "macro_average"],
        value_vars=TASKS,
        var_name="task",
        value_name="accuracy",
    )
    long["task_label"] = long["task"].map(TASK_LABELS)
    long["accuracy_pct"] = long["accuracy"] * 100
    return long


def to_long_gains(gains: pd.DataFrame) -> pd.DataFrame:
    """Convert wide DoRA-minus-LoRA gains to task-level long form."""
    long = gains.melt(
        id_vars=["scope", "scope_label", "macro_average"],
        value_vars=TASKS,
        var_name="task",
        value_name="delta",
    )
    long["task_label"] = long["task"].map(TASK_LABELS)
    long["delta_points"] = long["delta"] * 100
    long["macro_delta_points"] = long["macro_average"] * 100
    return long


def build_scope_summary(df: pd.DataFrame, gains: pd.DataFrame) -> pd.DataFrame:
    """Summarize macro gains, task win counts, and cross-task stability."""
    long_gains = to_long_gains(gains)
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        lora_row = _row_for(df, "lora", scope)
        dora_row = _row_for(df, "dora", scope)
        scope_gains = long_gains[long_gains["scope"] == scope]
        if lora_row.empty or dora_row.empty or scope_gains.empty:
            continue
        rows.append({
            "scope": scope,
            "scope_label": SCOPE_LABELS[scope],
            "lora_macro": _first_float(lora_row, "macro_average"),
            "dora_macro": _first_float(dora_row, "macro_average"),
            "delta": _first_float(gains[gains["scope"] == scope], "macro_average"),
            "delta_points": _first_float(gains[gains["scope"] == scope], "macro_average") * 100,
            "task_wins": int((scope_gains["delta"] > 0).sum()),
            "task_ties": int((scope_gains["delta"] == 0).sum()),
            "task_losses": int((scope_gains["delta"] < 0).sum()),
            "best_task": str(scope_gains.loc[scope_gains["delta"].idxmax(), "task_label"]),
            "worst_task": str(scope_gains.loc[scope_gains["delta"].idxmin(), "task_label"]),
            "mean_task_delta_points": float(scope_gains["delta_points"].mean()),
            "median_task_delta_points": float(scope_gains["delta_points"].median()),
            "task_delta_std_points": float(scope_gains["delta_points"].std(ddof=0)),
        })
    return pd.DataFrame(rows)


def build_task_summary(gains: pd.DataFrame) -> pd.DataFrame:
    """Summarize which scope gives DoRA the strongest per-task gain."""
    long_gains = to_long_gains(gains)
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = long_gains[long_gains["task"] == task]
        best = task_rows.loc[task_rows["delta"].idxmax()]
        worst = task_rows.loc[task_rows["delta"].idxmin()]
        rows.append({
            "task": task,
            "task_label": TASK_LABELS[task],
            "best_scope": best["scope"],
            "best_scope_label": best["scope_label"],
            "best_delta_points": float(best["delta_points"]),
            "worst_scope": worst["scope"],
            "worst_scope_label": worst["scope_label"],
            "worst_delta_points": float(worst["delta_points"]),
            "positive_scopes": int((task_rows["delta"] > 0).sum()),
        })
    return pd.DataFrame(rows)


def build_official_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare our full-scope macro result with the official LLaMA2-7B reference."""
    full_rows = df[df["scope"] == "full"].copy()
    ours = full_rows[["method", "macro_average", *TASKS]].copy()
    ours["source"] = "Reproduction"
    ours["rank"] = ours["method"].map({"lora": 32, "dora": 16})
    ours["params_percent"] = pd.NA
    ours["paper_method"] = ours["method"].map({"lora": "LoRA", "dora": "DoRA"})
    official = pd.DataFrame(OFFICIAL_LLAMA2_7B_REFERENCE)
    comparison = pd.concat([official, ours], ignore_index=True)
    comparison["method_label"] = comparison["method"].map(METHOD_LABELS)
    comparison["macro_points"] = comparison["macro_average"] * 100
    comparison["series"] = comparison["source"] + " - " + comparison["method_label"]
    return comparison[
        [
            "source",
            "method",
            "method_label",
            "paper_method",
            "rank",
            "params_percent",
            "macro_average",
            "macro_points",
            *TASKS,
        ]
    ]


def fig1_macro_grouped(df: pd.DataFrame, out: Path) -> None:
    """Grouped Seaborn bar chart of macro-average accuracy by method and scope."""
    plot_df = df.copy()
    plot_df["macro_points"] = plot_df["macro_average"] * 100
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    sns.barplot(
        data=plot_df,
        x="scope_label",
        y="macro_points",
        hue="method_label",
        hue_order=["LoRA", "DoRA"],
        palette=PALETTE,
        saturation=1.0,
        edgecolor="white",
        linewidth=1.0,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(
            cast("BarContainer", container),
            fmt="%.1f",
            padding=4,
            fontsize=13,
            fontweight="bold",
        )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Macro-average accuracy (%)")
    ax.set_ylim(72, 82.5)
    ax.legend(title="", frameon=True, loc="upper right")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    _save(fig, out)


def fig2_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Seaborn heatmap of per-task accuracy for all available conditions."""
    long_df = to_long_accuracy(df)
    heatmap_df = long_df.pivot_table(
        index="condition", columns="task_label", values="accuracy_pct", aggfunc="first"
    )
    heatmap_df = heatmap_df[[TASK_LABELS[task] for task in TASKS]]
    fig, (ax, cax) = plt.subplots(
        ncols=2,
        figsize=(15.8, 6.3),
        gridspec_kw={"width_ratios": [35, 1], "wspace": 0.05},
    )
    sns.heatmap(
        heatmap_df,
        cmap=sns.light_palette(DORAEMON_BLUE_DARK, as_cmap=True),
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 10.5, "fontweight": "bold"},
        linewidths=0.6,
        linecolor="white",
        cbar_ax=cax,
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
    )
    ax.set_title("Per-Task Accuracy Across All Conditions")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=28, labelsize=12)
    ax.tick_params(axis="y", rotation=0, labelsize=12)
    cax.tick_params(labelsize=11)
    cax.yaxis.label.set_size(13)
    _save(fig, out, tight=False)


def fig3_dora_gains(gains: pd.DataFrame, out: Path) -> None:
    """Diverging Seaborn heatmap of DoRA-minus-LoRA gains."""
    long_gains = to_long_gains(gains)
    heatmap_df = long_gains.pivot_table(
        index="scope_label", columns="task_label", values="delta_points", aggfunc="first"
    )
    heatmap_df = heatmap_df[[TASK_LABELS[task] for task in TASKS]]
    cmap = LinearSegmentedColormap.from_list(
        "lora_dora_delta",
        [LORAX_ORANGE, "white", DORAEMON_BLUE_DARK],
    )
    fig, (ax, cax) = plt.subplots(
        ncols=2,
        figsize=(15.8, 5.3),
        gridspec_kw={"width_ratios": [35, 1], "wspace": 0.05},
    )
    sns.heatmap(
        heatmap_df,
        cmap=cmap,
        center=0,
        annot=True,
        fmt="+.1f",
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        linewidths=0.6,
        linecolor="white",
        cbar_ax=cax,
        cbar_kws={"label": "DoRA - LoRA accuracy points"},
        ax=ax,
    )
    ax.set_title("Task-Level DoRA Gains by Adapter Scope")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=28, labelsize=13)
    ax.tick_params(axis="y", rotation=0, labelsize=13)
    cax.tick_params(labelsize=11)
    cax.yaxis.label.set_size(13)
    _save(fig, out, tight=False)


def fig4_delta_distribution(gains: pd.DataFrame, out: Path) -> None:
    """Show whether each scope's DoRA gain is broad or task-specific."""
    plot_df = to_long_gains(gains)
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    sns.boxplot(
        data=plot_df,
        x="scope_label",
        y="delta_points",
        color=LORAX_ORANGE_LIGHT,
        width=0.48,
        fliersize=0,
        linewidth=1.35,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="scope_label",
        y="delta_points",
        hue="task_label",
        palette="colorblind",
        size=6.5,
        jitter=0.22,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    macro_points = gains.set_index("scope_label")["macro_average"] * 100
    for index, scope_label in enumerate([SCOPE_LABELS[scope] for scope in SCOPES]):
        if scope_label in macro_points:
            ax.scatter(
                index, macro_points[scope_label], marker="D", s=110, color=DARK_TEXT, zorder=5
            )
            ax.text(
                index + 0.08,
                macro_points[scope_label],
                "macro",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
    ax.axhline(0, color=DARK_TEXT, linewidth=1.4)
    ax.set_title("Distribution of Task-Level DoRA Gains")
    ax.set_xlabel("")
    ax.set_ylabel("DoRA - LoRA accuracy points")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(title="Task", bbox_to_anchor=(1.01, 1.0), loc="upper left", frameon=True)
    sns.despine(ax=ax)
    _save(fig, out)


def fig5_official_comparison(comparison: pd.DataFrame, out: Path) -> None:
    """Compare full-scope reproduction macro accuracy with the official reference."""
    plot_df = comparison.copy()
    plot_df["label"] = plot_df["source"].str.replace(" Table 1 ", "\nTable 1 ", regex=False)
    plot_df["method_label"] = plot_df["method_label"].map({
        "LoRA": "LoRA (r=32)",
        "DoRA": "DoRA (r=16)",
    })
    palette = {
        "LoRA (r=32)": LORAX_ORANGE,
        "DoRA (r=16)": DORAEMON_BLUE,
    }
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    sns.barplot(
        data=plot_df,
        x="label",
        y="macro_points",
        hue="method_label",
        hue_order=["LoRA (r=32)", "DoRA (r=16)"],
        palette=palette,
        saturation=1.0,
        edgecolor="white",
        linewidth=1.0,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(
            cast("BarContainer", container),
            fmt="%.1f",
            padding=4,
            fontsize=20,
            fontweight="bold",
        )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Macro-average accuracy (%)")
    ax.set_ylim(72, 83)
    ax.legend(title="", frameon=True, loc="upper right")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    _save(fig, out)


def load_ranked_results() -> pd.DataFrame:
    """Load the checked-in rank sweep for the full-scope LoRA and DoRA runs."""
    if not CHECKED_IN_RANKED_PATH.exists():
        msg = f"Missing ranked summary CSV: {CHECKED_IN_RANKED_PATH}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(CHECKED_IN_RANKED_PATH)
    required = {"method", "scope", "Rank", "macro_average"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Ranked summary is missing required columns: {sorted(missing)!r}"
        raise ValueError(msg)
    df = df[(df["scope"] == "full") & (df["method"].isin(METHODS))].copy()
    df["method_label"] = df["method"].map(METHOD_LABELS)
    df["rank"] = df["Rank"].astype(int)
    df["macro_points"] = df["macro_average"].astype(float) * 100
    return df.sort_values(["method", "rank"]).reset_index(drop=True)


def fig6_rank_scaling(ranked_df: pd.DataFrame, out: Path) -> None:
    """Show how LoRA and DoRA scale with rank on the full-scope benchmark."""
    plot_df = ranked_df.copy()
    palette = {"LoRA": LORAX_ORANGE, "DoRA": DORAEMON_BLUE}

    fig, ax_top = plt.subplots(figsize=(14.8, 6.9))

    sns.lineplot(
        data=plot_df,
        x="rank",
        y="macro_points",
        hue="method_label",
        hue_order=["LoRA", "DoRA"],
        palette=palette,
        marker="o",
        markersize=14,
        dashes=False,
        linewidth=4.4,
        ax=ax_top,
    )
    for method_label in ["LoRA", "DoRA"]:
        method_df = plot_df[plot_df["method_label"] == method_label]
        for row in method_df.itertuples(index=False):
            ax_top.annotate(
                f"{row.macro_points:.1f}",
                (row.rank, row.macro_points),
                textcoords="offset points",
                xytext=(0, 12 if method_label == "LoRA" else -24),
                ha="center",
                va="bottom" if method_label == "LoRA" else "top",
                fontsize=18,
                fontweight="bold",
                color=palette[method_label],
            )

    ax_top.set_ylabel("Macro-average accuracy (%)", fontsize=19)
    ax_top.set_xlabel("Ranks", fontsize=19)
    ax_top.set_xscale("log", base=2)
    ax_top.set_xlim(3.5, 36)
    ax_top.set_xticks([4, 8, 16, 32])
    ax_top.set_xticklabels(["4", "8", "16", "32"])
    ax_top.set_ylim(75.5, 80.6)
    ax_top.legend(title="", frameon=True, loc="upper right")
    ax_top.tick_params(axis="both", labelsize=19)
    ax_top.grid(axis="x", visible=False)
    ax_top.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.7)
    sns.despine(ax=ax_top)

    _save(fig, out, dpi=220)


def print_summary(scope_summary: pd.DataFrame, task_summary: pd.DataFrame) -> None:
    """Print a compact console summary of the key analytical conclusions."""
    print("\n" + "=" * 78)
    print("  RESULT ANALYSIS: Macro-Average Accuracy and Task-Level Consistency")
    print("=" * 78)
    for row in cast("list[ScopeSummaryRecord]", scope_summary.to_dict("records")):
        print(
            f"  {row['scope_label']:<15} LoRA={row['lora_macro']:.4f}  "
            f"DoRA={row['dora_macro']:.4f}  delta={row['delta_points']:+.2f} pts  "
            f"wins/ties/losses={row['task_wins']}/{row['task_ties']}/{row['task_losses']}"
        )
        print(f"      best task: {row['best_task']}; weakest task: {row['worst_task']}")

    best_scope = scope_summary.loc[scope_summary["delta_points"].idxmax()]
    robust_scope = scope_summary.loc[scope_summary["task_wins"].idxmax()]
    print("\n  Interpretation")
    print("  " + "-" * 72)
    print(
        f"  Strongest macro reproduction: {best_scope['scope_label']} "
        f"({best_scope['delta_points']:+.2f} accuracy points)."
    )
    print(
        f"  Broadest task coverage: {robust_scope['scope_label']} "
        f"({robust_scope['task_wins']} positive task deltas out of {len(TASKS)})."
    )
    positive_tasks = task_summary[task_summary["positive_scopes"] > 0]
    print(
        f"  DoRA improves at least one scope on {len(positive_tasks)}/{len(TASKS)} tasks; "
        "attention-only is the least reliable isolation in these runs."
    )
    print()


def write_tables(
    out_dir: Path,
    df: pd.DataFrame,
    gains: pd.DataFrame,
    scope_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    official_comparison: pd.DataFrame,
) -> None:
    """Write all analysis tables used by the README/report."""
    df[["method", "scope", "macro_average", *TASKS]].to_csv(
        out_dir / "summary_table.csv", index=False
    )
    gains[["scope", *TASKS, "macro_average"]].to_csv(out_dir / "dora_gains_table.csv", index=False)
    scope_summary.to_csv(out_dir / "scope_summary.csv", index=False)
    task_summary.to_csv(out_dir / "task_summary.csv", index=False)
    official_comparison.to_csv(out_dir / "official_reference_comparison.csv", index=False)
    print("  Saved: summary_table.csv")
    print("  Saved: dora_gains_table.csv")
    print("  Saved: scope_summary.csv")
    print("  Saved: task_summary.csv")
    print("  Saved: official_reference_comparison.csv")


def generate_figures(
    out_dir: Path,
    df: pd.DataFrame,
    gains: pd.DataFrame,
    official_comparison: pd.DataFrame,
    ranked_df: pd.DataFrame,
) -> None:
    """Generate the Seaborn figure suite."""
    fig1_macro_grouped(df, out_dir / "fig1_macro_grouped.png")
    fig2_heatmap(df, out_dir / "fig2_per_task_heatmap.png")
    fig3_dora_gains(gains, out_dir / "fig3_dora_gains.png")
    fig4_delta_distribution(gains, out_dir / "fig4_delta_distribution.png")
    fig5_official_comparison(official_comparison, out_dir / "fig5_official_comparison.png")
    fig6_rank_scaling(ranked_df, out_dir / "fig6_rank_scaling.png")


def main() -> None:
    """Run the analysis script."""
    parser = argparse.ArgumentParser(description="Analyze DoRA vs LoRA results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to results/runs/ containing */metrics.json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis_output"),
        help="Directory for figures and CSV tables.",
    )
    args = parser.parse_args()
    _configure_plot_style()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nLoading results...")
    df = load_results(args.results)
    gains = compute_gains(df)
    scope_summary = build_scope_summary(df, gains)
    task_summary = build_task_summary(gains)
    official_comparison = build_official_comparison(df)
    ranked_df = load_ranked_results()
    print("\nWriting analysis tables...")
    write_tables(out_dir, df, gains, scope_summary, task_summary, official_comparison)
    print("\nGenerating Seaborn figures...")
    generate_figures(out_dir, df, gains, official_comparison, ranked_df)
    print_summary(scope_summary, task_summary)
    print(f"All outputs written to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
