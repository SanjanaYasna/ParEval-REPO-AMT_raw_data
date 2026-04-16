#!/usr/bin/env python3
"""
dumbbell_plot.py

Reads <model>_regular.csv and <model>_code_only.csv for codex and opus,
produces a dumbbell (connected dot) plot comparing Regular vs Code Only
for build@1, pass@3, run@1.

Usage:
    python dumbbell_plot.py /Users/robsonlab/scratch/hpx_to_legion_repo/
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

REPO_PROMPTS = ["Barnes_Hut", "AsyncSTM", "heapify", "osc_chain_1d"]

# Only the two models with non-zero @k values
MODEL_NAMES = ["gpt-5.3-codex", "opus"]

MODEL_DISPLAY_NAMES = {
    "gpt-5.3-codex": "GPT-5.3 Codex",
    "opus":          "Claude Opus 4.6",
}

MODEL_COLORS = {
    "gpt-5.3-codex": "#2a7fbf",
    "opus":          "#d94f4f",
}

METRICS = ["build@1", "pass@3", "run@1"]

METRIC_DISPLAY = {
    "build@1": "build@1",
    "pass@3":  "strict_pass@3",
    "run@1":   "run@1",
}

BASE_BG    = "#ffffff"
LABEL_COLOR = "#222222"
TICK_COLOR  = "#1a1a1a"
GRID_COLOR  = "#d0d0d0"

REG_MARKER   = "o"   # circle = regular
CO_MARKER    = "D"   # diamond = code_only



# ── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Dumbbell plot: Regular vs Code-Only @k metrics."
    )
    parser.add_argument("root_dir", type=str)
    parser.add_argument("-o", "--output-dir", type=str, default=".")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


# ── Data ingestion ───────────────────────────────────────────────────────────

def gather_data(root_dir: str):
    """Read <root>/<repo>/<model>/<model>_{regular,code_only}.csv."""
    reg_rows, co_rows = [], []

    for repo in REPO_PROMPTS:
        for model in MODEL_NAMES:
            base = os.path.join(root_dir, repo, model)
            reg_path = os.path.join(base, f"{model}_regular.csv")
            co_path  = os.path.join(base, f"{model}_code_only.csv")

            if os.path.isfile(reg_path):
                df = pd.read_csv(reg_path)
                df["repo"], df["model_key"] = repo, model
                reg_rows.append(df)
            if os.path.isfile(co_path):
                df = pd.read_csv(co_path)
                df["repo"], df["model_key"] = repo, model
                co_rows.append(df)

    regular   = pd.concat(reg_rows,  ignore_index=True) if reg_rows  else pd.DataFrame()
    code_only = pd.concat(co_rows,   ignore_index=True) if co_rows   else pd.DataFrame()
    print(f"Loaded {len(reg_rows)} regular + {len(co_rows)} code_only CSVs.")
    return regular, code_only


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_dumbbell(regular: pd.DataFrame, code_only: pd.DataFrame,
                  output_dir: str, dpi: int):
    """One column of panels, one per metric. Each row = repo, each model = colour."""

    n_metrics = len(METRICS)
    n_repos   = len(REPO_PROMPTS)
    n_models  = len(MODEL_NAMES)

    row_gap = 1.0
    sub_gap = 0.35

    fig, axes = plt.subplots(
        n_metrics, 1,
        figsize=(9, n_metrics * 3.8),
        constrained_layout=True,
    )
    fig.set_facecolor(BASE_BG)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        ax.set_facecolor(BASE_BG)

        # Build pivot tables: repo × model
        reg_piv = (regular.pivot_table(index="repo", columns="model_key",
                                       values=metric, aggfunc="mean")
                   .reindex(index=REPO_PROMPTS, columns=MODEL_NAMES).fillna(0))

        co_piv = (code_only.pivot_table(index="repo", columns="model_key",
                                        values=metric, aggfunc="mean")
                  .reindex(index=REPO_PROMPTS, columns=MODEL_NAMES).fillna(0))

        # ── Compute y positions ──────────────────────────────────────
        y_ticks_model  = []
        y_labels_model = []
        y_ticks_repo   = []

        y = 0

        for repo_idx, repo in enumerate(REPO_PROMPTS):
            group_start = y

            for model_idx, model in enumerate(MODEL_NAMES):
                reg_val = reg_piv.loc[repo, model]
                co_val  = co_piv.loc[repo, model]
                color   = MODEL_COLORS[model]

                # Connecting line
                ax.plot(
                    [reg_val, co_val], [y, y],
                    color=color, linewidth=2.2, alpha=0.5, zorder=1,
                )

                # Regular dot
                ax.scatter(
                    reg_val, y,
                    color=color, marker=REG_MARKER,
                    s=90, zorder=3, edgecolors="white", linewidths=0.8,
                )
                # Code-only dot
                ax.scatter(
                    co_val, y,
                    color=color, marker=CO_MARKER,
                    s=90, zorder=3, edgecolors="white", linewidths=0.8,
                )

                # Value labels: regular to the LEFT, code_only to the RIGHT
                ax.annotate(
                    f"{reg_val:.2f}",
                    xy=(reg_val, y),
                    xytext=(-14, 0),
                    textcoords="offset points",
                    fontsize=8, fontweight="bold",
                    color=color, ha="right", va="center",
                )
                ax.annotate(
                    f"{co_val:.2f}",
                    xy=(co_val, y),
                    xytext=(14, 0),
                    textcoords="offset points",
                    fontsize=8, fontweight="bold",
                    color=color, ha="left", va="center",
                )

                # Delta label at midpoint
                delta = co_val - reg_val
                if abs(delta) > 0.005:
                    mid_x = (reg_val + co_val) / 2
                    sign  = "+" if delta > 0 else ""
                    ax.text(
                        mid_x, y - sub_gap * 0.6,
                        f"{sign}{delta:.2f}",
                        ha="center", va="top", fontsize=6.5,
                        color="#555555", style="italic",
                    )

                y_ticks_model.append(y)
                y_labels_model.append(f"  {MODEL_DISPLAY_NAMES[model]}")

                y += sub_gap

            # Repo label: centered vertically on its group
            group_center = (group_start + y - sub_gap) / 2
            y_ticks_repo.append(group_center)

            y += row_gap

        # ── Alternating row shading ──────────────────────────────────
        shade_y = 0
        for repo_idx in range(n_repos):
            group_bottom = shade_y - sub_gap * 0.5
            group_top    = shade_y + (n_models - 1) * sub_gap + sub_gap * 0.5
            if repo_idx % 2 == 0:
                ax.axhspan(group_bottom, group_top,
                           color="#f4f4f8", zorder=0)
            shade_y += n_models * sub_gap + row_gap

        # ── Y-axis: model names as tick labels ───────────────────────
        ax.set_yticks(y_ticks_model)
        ax.set_yticklabels(y_labels_model, fontsize=8)

        # ── Y-axis: repo names as bold text, centered on each group ──
        for center_y, repo in zip(y_ticks_repo, REPO_PROMPTS):
            # ax.text(
            #     -0.08, center_y,
            #     repo,
            #     transform=ax.get_yaxis_transform(),
            #     ha="right", va="center",
            #     fontsize=9.5, fontweight="bold",
            #     color=LABEL_COLOR,
            # )
            ax.text(
                -0.2, center_y,
                repo,
                transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=9.5, fontweight="bold",
                color=LABEL_COLOR,
                rotation=45,
            )
            

        # ── Axis styling ─────────────────────────────────────────────
        ax.set_xlim(-0.18, 1.18)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel(METRIC_DISPLAY[metric], fontsize=11,
                      fontweight="semibold", color=LABEL_COLOR)
        ax.set_title(METRIC_DISPLAY[metric], fontsize=13,
                     fontweight="bold", color=LABEL_COLOR, pad=10)
        ax.invert_yaxis()

        ax.grid(axis="x", color=GRID_COLOR, linestyle="--",
                linewidth=0.6, alpha=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#b0b0b0")
        ax.spines["bottom"].set_color("#b0b0b0")
        ax.tick_params(axis="both", colors=TICK_COLOR, labelsize=9)

    # ── Legend ────────────────────────────────────────────────────────
    legend_elements = []
    for model in MODEL_NAMES:
        legend_elements.append(
            mlines.Line2D(
                [], [], color=MODEL_COLORS[model],
                marker=REG_MARKER, markersize=8, linestyle="None",
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"{MODEL_DISPLAY_NAMES[model]} — Full Original ",
            )
        )
        legend_elements.append(
            mlines.Line2D(
                [], [], color=MODEL_COLORS[model],
                marker=CO_MARKER, markersize=8, linestyle="None",
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"{MODEL_DISPLAY_NAMES[model]} — Code-Only",
            )
        )
    legend_elements.append(
        mlines.Line2D(
            [], [], color="#888888", linewidth=2, alpha=0.5,
            label="Δ (Full Original → Code-Only Improvement)",
        )
    )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        fontsize=9,
        frameon=True, framealpha=0.95,
        edgecolor="#b7b7b7",
        facecolor="#f9f9f9",
    )

    out_path = os.path.join(output_dir, "dumbbell_regular_vs_code_only.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=BASE_BG)
    print(f"Figure saved to {out_path}")
    plt.show()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    args = get_args()
    root_dir = os.path.abspath(args.root_dir)

    if not os.path.isdir(root_dir):
        print(f"ERROR: {root_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    regular, code_only = gather_data(root_dir)

    if regular.empty and code_only.empty:
        print("No CSV files found. Expected:", file=sys.stderr)
        print("  <root>/<repo>/<model>/<model>_{regular,code_only}.csv", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_dumbbell(regular, code_only, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()