#!/usr/bin/env python3
"""
atk_comparison.py

Finds <model>_regular.csv and <model>_code_only.csv files under
    <root>/<repository>/<model>/

Produces side-by-side horizontal bar charts (Regular | Code Only)
for metrics: build@1, pass@3, run@1.

Usage:
    python atk_comparison.py /Users/robsonlab/scratch/hpx_to_legion_repo/
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
from matplotlib.patches import Patch
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

REPO_PROMPTS = ["Barnes_Hut", "AsyncSTM", "heapify", "osc_chain_1d"]

MODEL_NAMES = [
    "gpt-5.3-codex", "glm", "hpccoder", "magicoder", "oss", "minimax", "opus",
]

MODEL_DISPLAY_NAMES = {
    "gpt-5.3-codex": "GPT-5.3 Codex",
    "glm":           "GLM",
    "hpccoder":      "HPCCoder",
    "magicoder":     "Magicoder",
    "oss":           "OSS",
    "minimax":       "MiniMax",
    "opus":          "Opus",
}

METRICS = ["build@1", "pass@3", "run@1"]

BASE_BG    = "#ffffff"
TICK_COLOR = "#1a1a1a"
GRID_COLOR = "#d0d0d0"
LABEL_COLOR = "#222222"
MIN_VISUAL_WIDTH = 0.02

MODEL_COLORS = [
    "#87bff2", "#f4a7b0", "#9ed7a5", "#f8d996",
    "#c5b7f5", "#ffcab1", "#a9d4ff",
]
MODEL_OUTLINES = [
    "#2a5d90", "#a22e50", "#2c7c53", "#a26a18",
    "#4d3b99", "#a14c3d", "#2f5f84",
]
MODEL_HATCHES = ["\\\\", "//", "oo", "++", "..", "xx", "||"]


# ── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Side-by-side Regular vs Code-Only @k bar charts."
    )
    parser.add_argument("root_dir", type=str)
    parser.add_argument("-o", "--output-dir", type=str, default=".")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


# ── Data ingestion ───────────────────────────────────────────────────────────

def gather_data(root_dir: str):
    """Directly check <root>/<repo>/<model>/<model>_{regular,code_only}.csv."""
    regular_rows, code_only_rows = [], []

    for repo in REPO_PROMPTS:
        for model in MODEL_NAMES:
            base = os.path.join(root_dir, repo, model)
            reg_path = os.path.join(base, f"{model}_regular.csv")
            co_path  = os.path.join(base, f"{model}_code_only.csv")

            if os.path.isfile(reg_path):
                df = pd.read_csv(reg_path)
                df["repo"], df["model_key"] = repo, model
                regular_rows.append(df)

            if os.path.isfile(co_path):
                df = pd.read_csv(co_path)
                df["repo"], df["model_key"] = repo, model
                code_only_rows.append(df)

    print(f"Found {len(regular_rows)} regular + {len(code_only_rows)} code_only CSVs.")
    regular   = pd.concat(regular_rows,   ignore_index=True) if regular_rows   else pd.DataFrame()
    code_only = pd.concat(code_only_rows, ignore_index=True) if code_only_rows else pd.DataFrame()
    return regular, code_only


# ── Styling helpers ──────────────────────────────────────────────────────────

def _bar_style(idx: int) -> dict:
    return {
        "color":     mcolors.to_rgba(MODEL_COLORS[idx % len(MODEL_COLORS)], alpha=0.7),
        "edgecolor": MODEL_OUTLINES[idx % len(MODEL_OUTLINES)],
        "linewidth": 1.8,
        "hatch":     MODEL_HATCHES[idx % len(MODEL_HATCHES)],
        "alpha":     0.85,
        "path_effects": [
            patheffects.withStroke(linewidth=2.4, foreground="#00000088")
        ],
    }


def _build_legend_handles():
    return [
        Patch(
            facecolor=mcolors.to_rgba(MODEL_COLORS[i % len(MODEL_COLORS)], alpha=0.7),
            edgecolor=MODEL_OUTLINES[i % len(MODEL_OUTLINES)],
            linewidth=1.8,
            hatch=MODEL_HATCHES[i % len(MODEL_HATCHES)],
            label=MODEL_DISPLAY_NAMES.get(m, m),
        )
        for i, m in enumerate(MODEL_NAMES)
    ]


# ── Panel drawing ────────────────────────────────────────────────────────────

def _draw_panel(ax, pivot, panel_title: str):
    """Draw grouped horizontal bars for one panel (Regular or Code Only)."""
    num_repos  = len(REPO_PROMPTS)
    num_models = len(MODEL_NAMES)
    bar_height = 0.75 / max(num_models, 1)
    pad = 0.03

    y_pos = np.arange(num_repos)

    for idx, model in enumerate(MODEL_NAMES):
        offsets = y_pos + (idx - (num_models - 1) / 2) * bar_height
        widths  = pivot[model].to_numpy()
        vis_w   = np.clip(widths, MIN_VISUAL_WIDTH, 1.0)

        bars = ax.barh(offsets, vis_w, height=bar_height, **_bar_style(idx))

        # Glass-highlight strip
        hw = vis_w * 0.15
        ax.barh(offsets, hw, height=bar_height, left=vis_w - hw,
                color="#ffffff", alpha=0.15, edgecolor="none")

        # Value labels
        for bar, true_val in zip(bars, widths):
            ax.annotate(
                f"{true_val:.2f}",
                xy=(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2),
                ha="left", va="center", fontsize=8, color="#000000",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(REPO_PROMPTS, fontsize=10)
    ax.set_xlim(0, 1.0 + pad * 5)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(panel_title, fontsize=12, fontweight="semibold", color=LABEL_COLOR)
    ax.tick_params(axis="both", labelsize=10, colors=TICK_COLOR)

    ax.set_facecolor(BASE_BG)
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color("#6b6f7e")
        spine.set_linewidth(0.9)

    ax.axvline(x=1.0, color="#ff6b6b", linestyle="--",
               linewidth=1.2, alpha=0.6, zorder=0)


# ── Main plotting ────────────────────────────────────────────────────────────

def plot_comparison(regular: pd.DataFrame, code_only: pd.DataFrame,
                    output_dir: str, dpi: int):
    """One figure: 3 metric-rows × 2 columns (Regular | Code Only)."""
    nrows = len(METRICS)
    fig, axes = plt.subplots(nrows, 2, figsize=(16, nrows * 4.2),
                             constrained_layout=False)
    fig.set_facecolor(BASE_BG)

    for row, metric in enumerate(METRICS):
        # Pivot: repos × models
        reg_piv = (regular.pivot_table(index="repo", columns="model_key",
                                       values=metric, aggfunc="mean")
                   .reindex(index=REPO_PROMPTS, columns=MODEL_NAMES).fillna(0))

        co_piv  = (code_only.pivot_table(index="repo", columns="model_key",
                                         values=metric, aggfunc="mean")
                   .reindex(index=REPO_PROMPTS, columns=MODEL_NAMES).fillna(0))

        _draw_panel(axes[row, 0], reg_piv,  f"Regular — {metric}")
        _draw_panel(axes[row, 1], co_piv,   f"Code Only — {metric}")

    # Top legend with display names
    handles = _build_legend_handles()
    legend = fig.legend(
        handles=handles, title="Model",
        loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=min(len(handles), 4),
        frameon=True, framealpha=0.95, fontsize=10,
        handlelength=1.6, handleheight=1.2, columnspacing=1.0,
    )
    legend.get_title().set_color(LABEL_COLOR)
    legend.get_frame().set_facecolor("#f9f9f9")
    legend.get_frame().set_edgecolor("#b7b7b7")

    fig.subplots_adjust(top=0.91, bottom=0.04, hspace=0.42, wspace=0.28)

    out_path = os.path.join(output_dir, "atk_regular_vs_code_only.png")
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
        print("No CSV files found. Expected layout:", file=sys.stderr)
        print("  <root>/<repo>/<model>/<model>_regular.csv", file=sys.stderr)
        print("  <root>/<repo>/<model>/<model>_code_only.csv", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_comparison(regular, code_only, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()