#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time depth metric calculation module
Purpose: compute metrics related to time depth
- Max, Min, Min_nonzero, Pq_nonzero
- AncientRatio
- TimeDepthKernel
- KernelScore (multiple time scales)
"""

from pathlib import Path
from typing import Union, List, Optional, Dict

import argparse
import math
import numpy as np
import pandas as pd


def _format_scale_label(scale: float) -> str:
    """Format a time-scale label for use as a column name"""
    text = f"{scale:g}"
    text = text.replace(".", "p").replace("-", "m")
    return text


def _compute_kernel_scores(
    values: np.ndarray,
    scales: List[float],
    sigma: float,
    total_count: int,
) -> Dict[str, float]:
    """Compute kernel scores across multiple time scales"""
    if sigma <= 0:
        raise ValueError("kernel_sigma must be > 0")
    scores: Dict[str, float] = {}
    if total_count <= 0:
        return scores
    arr = np.asarray(values, dtype=float)
    positives = arr[arr > 0]
    if positives.size == 0:
        for s in scales:
            label = _format_scale_label(s)
            scores[f"KernelScore_T{label}"] = 0.0
        return scores

    log_vals = np.log10(positives)
    denom = float(total_count)
    for s in scales:
        if s is None or s <= 0:
            continue
        log_t = math.log(float(s))
        weights = np.exp(-((log_vals - log_t) ** 2) / (2.0 * sigma * sigma))
        score = float(weights.sum()) / denom
        scores[f"KernelScore_T{_format_scale_label(s)}"] = score
    return scores


def _compute_time_depth_kernel(
    values: np.ndarray,
    focus_years: float,
    sigma_log10: float,
) -> float:
    """Compute smoothed time depth index (using log10 time scale)"""
    if sigma_log10 <= 0 or focus_years <= 0:
        return 0.0
    arr = np.asarray(values, dtype=float)
    positives = arr[arr > 0]
    if positives.size == 0:
        return 0.0
    log_vals = np.log10(positives)
    log_focus = math.log10(focus_years)
    weights = np.exp(-((log_vals - log_focus) ** 2) / (2.0 * sigma_log10 * sigma_log10))
    return float(weights.mean())


def _parse_kernel_scales(arg: str) -> List[float]:
    """Parse a comma-separated list of time scales"""
    if not arg:
        return []
    out: List[float] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            raise ValueError(f"Failed to parse numeric value in kernel_scales: {token!r}")
    return out


def calc_time_depth_stats(
    df: pd.DataFrame,
    group_col: Optional[Union[str, List[str]]] = None,
    tmrca_col: str = "Time_years",
    ancient_threshold: float = 100000.0,
    ratio_quantile: float = 0.01,
    kernel_scales: Optional[List[float]] = None,
    kernel_sigma: float = 0.5,
    time_depth_focus: Optional[float] = None,
    time_depth_sigma_log10: float = 0.3,
) -> pd.DataFrame:
    """
    Compute metrics related to time depth
    
    Returns columns:
    - grouping columns
    - Count: sample size
    - Max: maximum TMRCA
    - Min: minimum TMRCA
    - Min_nonzero: minimum positive TMRCA
    - Pq_nonzero: ratio_quantile quantile of positive samples
    - AncientRatio(>threshold): proportion of ancient lineages
    - KernelScore_T*: kernel scores at each time scale
    - TimeDepthKernel_T*: smoothed time depth index
    """
    # Grouper
    if group_col is None:
        groups = [(None, df)]
        group_cols: Optional[List[str]] = None
    else:
        groups = df.groupby(group_col)
        group_cols = group_col if isinstance(group_col, (list, tuple)) else [group_col]

    results: List[Dict[str, object]] = []

    for name, g in groups:
        vals = pd.to_numeric(g[tmrca_col], errors="coerce").dropna()
        n = len(vals)
        if n < 2:
            continue

        # Basic statistics
        v_max = float(vals.max())
        v_min = float(vals.min())

        # Positive sample quantile
        positives = vals[vals > 0]
        if len(positives) > 0:
            p_nonzero = float(np.quantile(positives, ratio_quantile))
            if p_nonzero <= 0:
                p_nonzero = np.nan
            min_nonzero = float(positives.min())
        else:
            p_nonzero = np.nan
            min_nonzero = np.nan

        # Proportion of ancient lineages
        ancient_ratio = float((vals > ancient_threshold).mean())

        # Time-scale kernel scores
        kernel_scores: Dict[str, float] = {}
        if kernel_scales:
            kernel_scores = _compute_kernel_scores(vals.values, kernel_scales, kernel_sigma, n)

        # Smoothed time depth index
        td_focus = time_depth_focus or ancient_threshold
        if td_focus is not None and td_focus > 0:
            time_depth_kernel = _compute_time_depth_kernel(vals.values, float(td_focus), time_depth_sigma_log10)
        else:
            time_depth_kernel = 0.0

        # Build result row
        row: Dict[str, object] = {
            "Count": int(n),
            "Max": v_max,
            "Min": v_min,
            "Min_nonzero": min_nonzero,
            "Pq_nonzero": p_nonzero,
            f"AncientRatio(>{int(ancient_threshold)})": ancient_ratio,
        }
        row.update(kernel_scores)
        td_label = _format_scale_label(td_focus) if td_focus else "auto"
        row[f"TimeDepthKernel_T{td_label}"] = time_depth_kernel

        # Add grouping columns
        if group_cols is None:
            row["Group"] = "ALL"
        else:
            if not isinstance(name, tuple):
                name = (name,)
            for col, val in zip(group_cols, name):
                row[col] = val

        results.append(row)

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Column order: grouping columns first
    front_cols = (group_cols or ["Group"])
    metric_cols = [c for c in out.columns if c not in front_cols]
    return out[front_cols + metric_cols]


def _parse_group_col(arg: str) -> Optional[Union[str, List[str]]]:
    """Parse the group_col argument from the command line"""
    if arg is None:
        return "Continent"
    s = arg.strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip() != ""]
    return s


def run(
    csv_path: Path,
    out_csv: Optional[Path] = None,
    group_col: Optional[Union[str, List[str]]] = "Continent",
    tmrca_col: str = "Time_years",
    ancient_threshold: float = 100000.0,
    ratio_quantile: float = 0.01,
    kernel_scales: Optional[List[float]] = None,
    kernel_sigma: float = 0.5,
    time_depth_focus: Optional[float] = None,
    time_depth_sigma_log10: float = 0.3,
    print_result: bool = True,
) -> pd.DataFrame:
    """
    Execute time depth metric calculation.
    """
    try:
        from tabulate import tabulate
        have_tabulate = True
    except Exception:
        have_tabulate = False

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if not (0.0 < ratio_quantile < 1.0):
        raise ValueError(f"--ratio_quantile must be within (0,1): current {ratio_quantile}")

    if isinstance(group_col, (list, tuple)):
        parsed_group_col = list(group_col)
    elif group_col is None:
        parsed_group_col = None
    else:
        parsed_group_col = _parse_group_col(group_col)  # type: ignore[arg-type]
    df = pd.read_csv(csv_path)

    # Parse kernel_scales
    if kernel_scales is not None and kernel_scales != []:
        parsed_kernel_scales = kernel_scales
    else:
        t = float(ancient_threshold)
        parsed_kernel_scales = [t / 10.0, t, t * 10.0]

    # Calculation
    res = calc_time_depth_stats(
        df,
        group_col=parsed_group_col,
        tmrca_col=tmrca_col,
        ancient_threshold=float(ancient_threshold),
        ratio_quantile=float(ratio_quantile),
        kernel_scales=parsed_kernel_scales,
        kernel_sigma=float(kernel_sigma),
        time_depth_focus=float(time_depth_focus) if time_depth_focus else float(ancient_threshold),
        time_depth_sigma_log10=float(time_depth_sigma_log10),
    )

    # Print
    if print_result:
        if res.empty:
            print("Empty result (possibly too few samples or mismatched column names).")
        else:
            if have_tabulate:
                print(tabulate(res, headers="keys", tablefmt="github", showindex=False))
            else:
                print(res.to_string(index=False))

    # Write out
    output_csv = out_csv if out_csv is not None else csv_path.parent / "time_depth_stats.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if print_result:
        print(f"\nSaved results to: {output_csv.resolve()}")
    return res


def main():
    try:
        from tabulate import tabulate
        have_tabulate = True
    except Exception:
        have_tabulate = False

    parser = argparse.ArgumentParser(
        description="Compute time depth metrics (Max, Min, AncientRatio, TimeDepthKernel, KernelScore)"
    )
    parser.add_argument("--csv_path", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--out_csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--group_col", type=str, default="Continent",
                        help='Grouping columns: single column like "Continent", or multiple like "Continent,Country"; pass "none" for no grouping')
    parser.add_argument("--tmrca_col", type=str, default="Time_years", help="TMRCA value column name")
    parser.add_argument("--ancient_threshold", type=float, default=100000.0, help="AncientRatio threshold")
    parser.add_argument("--ratio_quantile", type=float, default=0.01, help="Positive-sample quantile (0-1) used as denominator")
    parser.add_argument("--kernel_scales", type=str, default="",
                        help="Comma-separated list of time scales (default automatically uses 0.1×, 1×, 10× of ancient_threshold)")
    parser.add_argument("--kernel_sigma", type=float, default=0.5,
                        help="Standard deviation of the Gaussian kernel on log scale σ")
    parser.add_argument("--time_depth_focus", type=float, default=None,
                        help="Focus T for the smoothed time depth metric (defaults to ancient_threshold)")
    parser.add_argument("--time_depth_sigma_log10", type=float, default=0.3,
                        help="Log10 standard deviation for the time depth Gaussian kernel")

    args = parser.parse_args()

    # Validation
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv_path}")
    if not (0.0 < args.ratio_quantile < 1.0):
        raise ValueError(f"--ratio_quantile must be within (0,1): current {args.ratio_quantile}")

    kernel_scales = _parse_kernel_scales(args.kernel_scales) if args.kernel_scales else None
    run(
        csv_path=args.csv_path,
        out_csv=args.out_csv,
        group_col=args.group_col,
        tmrca_col=args.tmrca_col,
        ancient_threshold=float(args.ancient_threshold),
        ratio_quantile=float(args.ratio_quantile),
        kernel_scales=kernel_scales,
        kernel_sigma=float(args.kernel_sigma),
        time_depth_focus=float(args.time_depth_focus) if args.time_depth_focus else None,
        time_depth_sigma_log10=float(args.time_depth_sigma_log10),
        print_result=True,
    )


if __name__ == "__main__":
    main()
