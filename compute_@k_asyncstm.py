#!/usr/bin/env python3
"""Compute build@k, pass@k, and run@k metrics from a single JSON results file."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from math import comb


# -----------------------------------------------------------------------------
# argument parsing
# -----------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        type=str,
        help="JSON file containing a list of result objects.",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="K values for build@k, pass@k, run@k (default: 1 3 5).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Base output CSV file path. Will produce _regular, _code_only, _combined, and _aggregate CSVs.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        default=["llm_name", "source_model", "dest_model"],
        help="Fields to group by when computing metrics (default: llm_name source_model dest_model).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def nCr(n: int, r: int) -> int:
    if n < r:
        return 1
    return comb(n, r)


def _passk(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased estimator of pass@k."""
    if num_samples < k:
        if num_samples == 0:
            return 0.0
        return float(num_correct > 0)
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))


def get_worst_status(status_dict: Dict[str, str]) -> str:
    """Extract the worst status from a dict of file statuses."""
    if not status_dict:
        return "MISSING"

    priority = {
        "INCORRECT_MODEL": 0,
        "BUILD_FAIL": 1,
        "RUNTIME_FAIL": 2,
        "MISSING": 3,
        "VALIDATION_FAIL": 4,
        "PASS": 5,
    }
    return min(status_dict.values(), key=lambda x: priority.get(x, 3))


def derive_flags(status: str) -> Dict[str, bool]:
    """
    From a status string, derive the three success flags:
      - did_build: not BUILD_FAIL and not INCORRECT_MODEL -> build@k
      - did_run:   VALIDATION_FAIL or PASS (MISSING COUNTS AGAINST IT) -> run@k
      - did_pass:  only PASS -> pass@k
    """
    return {
        "did_build": status not in ("BUILD_FAIL", "INCORRECT_MODEL"),
        "did_run": status in ("VALIDATION_FAIL", "PASS"),
        "did_pass": status == "PASS",
    }


def json_to_dataframes(data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert JSON entries to six DataFrames:
    - per-file resolution: reg_file, co_file, comb_file
    - aggregate (worst) resolution: reg_agg, co_agg, comb_agg
    """
    reg_file_rows = []
    co_file_rows = []
    reg_agg_rows = []
    co_agg_rows = []

    for entry in data:
        base_meta = {
            "app": entry.get("app", "unknown"),
            "prompt_strategy": entry.get("prompt_strategy", "unknown"),
            "llm_name": entry.get("llm_name", "unknown"),
            "source_model": entry.get("source_model", "unknown"),
            "dest_model": entry.get("dest_model", "unknown"),
            "output_number": entry.get("output_number", 0),
        }

        reg_dict = entry.get("overall_status", {})
        co_dict = entry.get("code_only_overall_status", {})

        # 1. Aggregate tracking (collapse multiple files to a single worst status per pipeline)
        r_agg = get_worst_status(reg_dict)
        c_agg = get_worst_status(co_dict)
        reg_agg_rows.append({**base_meta, "status": r_agg, **derive_flags(r_agg)})
        co_agg_rows.append({**base_meta, "status": c_agg, **derive_flags(c_agg)})

        # 2. Per-file tracking
        all_files = set(reg_dict.keys()).union(set(co_dict.keys()))
        if not all_files:
            all_files = {"unknown_file"}

        for f in all_files:
            r_stat = reg_dict.get(f, "MISSING")
            c_stat = co_dict.get(f, "MISSING")
            reg_file_rows.append({**base_meta, "file_name": f, "status": r_stat, **derive_flags(r_stat)})
            co_file_rows.append({**base_meta, "file_name": f, "status": c_stat, **derive_flags(c_stat)})

    df_reg_file = pd.DataFrame(reg_file_rows)
    df_co_file = pd.DataFrame(co_file_rows)
    df_comb_file = pd.concat([df_reg_file, df_co_file], ignore_index=True) if not df_reg_file.empty else pd.DataFrame()

    df_reg_agg = pd.DataFrame(reg_agg_rows)
    df_co_agg = pd.DataFrame(co_agg_rows)
    df_comb_agg = pd.concat([df_reg_agg, df_co_agg], ignore_index=True) if not df_reg_agg.empty else pd.DataFrame()

    return df_reg_file, df_co_file, df_comb_file, df_reg_agg, df_co_agg, df_comb_agg


ALL_STATUSES = [
    "INCORRECT_MODEL",
    "BUILD_FAIL",
    "RUNTIME_FAIL",
    "MISSING",
    "VALIDATION_FAIL",
    "PASS",
]


def compute_all_metrics(
    df: pd.DataFrame,
    k_values: List[int],
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Given a standardized DataFrame of generations (status, did_build, did_run, did_pass),
    compute the status summaries and @k metrics.
    """
    if df.empty:
        return pd.DataFrame()

    # 1. Status counts per group
    status_counts = df.groupby(group_cols + ["status"]).size().unstack(fill_value=0).reset_index()
    status_counts.columns.name = None

    # Ensure all possible status columns exist reliably every time
    for s in ALL_STATUSES:
        if s not in status_counts.columns:
            status_counts[s] = 0

    # Calculate total directly from counts
    status_counts["total_samples"] = status_counts[ALL_STATUSES].sum(axis=1)

    # 2. Metrics computation (computed per app, then averaged per group)
    agg_cols = ["app"] + group_cols
    app_agg = df.groupby(agg_cols).agg(
        num_samples=("status", "count"),
        num_build=("did_build", "sum"),
        num_run=("did_run", "sum"),
        num_pass=("did_pass", "sum"),
    ).reset_index()

    for k in k_values:
        app_agg[f"build@{k}"] = app_agg.apply(lambda x: _passk(x["num_samples"], x["num_build"], k), axis=1)
        app_agg[f"run@{k}"]   = app_agg.apply(lambda x: _passk(x["num_samples"], x["num_run"], k), axis=1)
        app_agg[f"pass@{k}"]  = app_agg.apply(lambda x: _passk(x["num_samples"], x["num_pass"], k), axis=1)

    metric_cols = [c for c in app_agg.columns if "@" in c]
    # Sort columns by metric type logically (build -> run -> pass) then by k
    order_map = {"build": 0, "run": 1, "pass": 2}
    metric_cols = sorted(metric_cols, key=lambda x: (order_map.get(x.split("@")[0], 99), int(x.split("@")[1])))

    # Final Aggregation Rules: average the @k metrics, track unique apps
    agg_dict = {c: (c, "mean") for c in metric_cols}
    agg_dict["apps"] = ("app", "nunique")

    metrics_df = app_agg.groupby(group_cols).agg(**agg_dict).reset_index()

    # 3. Merge status counts and metrics together
    result = pd.merge(status_counts, metrics_df, on=group_cols, how="outer")

    # 4. Reorder columns uniformly
    ordered_cols = group_cols + ["apps", "total_samples"] + ALL_STATUSES + metric_cols
    ordered_cols = [c for c in ordered_cols if c in result.columns]

    return result[ordered_cols]


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    args = get_args()
    input_file = Path(args.input_file).resolve()

    if not input_file.is_file():
        raise SystemExit(f"Error: {input_file} is not a file.")

    with input_file.open("r") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise SystemExit("Error: JSON file must contain a list of result objects.")

    if not data:
        raise SystemExit("Error: JSON file contains no entries.")

    print(f"Loaded {len(data)} entries from {input_file}")

    # Process JSON into structured standard pandas DataFrames
    df_reg_file, df_co_file, df_comb_file, df_reg_agg, df_co_agg, df_comb_agg = json_to_dataframes(data)

    # Validate base group-by columns exist
    for col in args.group_by:
        if col not in df_reg_file.columns:
            raise SystemExit(f"Error: group-by column '{col}' not found. Available: {list(df_reg_file.columns)}")

    # For the per-file pipelines, we add "file_name" to the group columns
    file_group_cols = args.group_by + ["file_name"]

    # Compute metrics for per-file pipelines
    res_reg_file = compute_all_metrics(df_reg_file, args.k, file_group_cols)
    res_co_file = compute_all_metrics(df_co_file, args.k, file_group_cols)
    res_comb_file = compute_all_metrics(df_comb_file, args.k, file_group_cols)

    # Compute metrics for aggregate (collapsed-file) pipelines
    res_reg_agg = compute_all_metrics(df_reg_agg, args.k, args.group_by)
    res_co_agg = compute_all_metrics(df_co_agg, args.k, args.group_by)
    res_comb_agg = compute_all_metrics(df_comb_agg, args.k, args.group_by)

    # Combine all 3 aggregated variants into a single 4th DataFrame for output
    if not res_reg_agg.empty: res_reg_agg.insert(0, "pipeline", "regular")
    if not res_co_agg.empty: res_co_agg.insert(0, "pipeline", "code_only")
    if not res_comb_agg.empty: res_comb_agg.insert(0, "pipeline", "combined")
    
    res_aggregate_all = pd.concat([res_reg_agg, res_co_agg, res_comb_agg], ignore_index=True)

    # Display Results
    print("\n=== Regular Pipeline (Split by File) ===")
    print(res_reg_file.to_string(index=False))

    print("\n=== Code-Only Pipeline (Split by File) ===")
    print(res_co_file.to_string(index=False))

    print("\n=== Combined Pool (Split by File) ===")
    print(res_comb_file.to_string(index=False))

    print("\n=== Aggregate Pipeline (All Files Collapsed) ===")
    print(res_aggregate_all.to_string(index=False))

    # Output CSVs
    if args.output:
        base_path = Path(args.output)
        stem = base_path.stem
        ext = base_path.suffix if base_path.suffix else ".csv"
        
        path_reg = base_path.with_name(f"{stem}_regular{ext}")
        path_co = base_path.with_name(f"{stem}_code_only{ext}")
        path_comb = base_path.with_name(f"{stem}_combined{ext}")
        path_agg = base_path.with_name(f"{stem}_aggregate{ext}")

        res_reg_file.to_csv(path_reg, index=False)
        res_co_file.to_csv(path_co, index=False)
        res_comb_file.to_csv(path_comb, index=False)
        res_aggregate_all.to_csv(path_agg, index=False)

        print(f"\nWrote CSVs to {base_path.parent}/")
        print(f"  - {path_reg.name}")
        print(f"  - {path_co.name}")
        print(f"  - {path_comb.name}")
        print(f"  - {path_agg.name}")


if __name__ == "__main__":
    main()