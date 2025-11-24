#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time distribution metric calculation module
Purpose: calculate metrics related to the temporal distribution structure
- Range (log10 scale)
- StdDev (log10 scale)
- Skewness (log10 scale, robust calculation)
- Estimated_Peaks (GMM + BIC)
- Ratio(Max/Pq_nonzero)
"""

from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict

import argparse
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew as moment_skew
from sklearn.mixture import GaussianMixture


def _bowley_skew(x: np.ndarray) -> float:
    """Bowley (quantile) skewness; returns 0.0 when IQR=0"""
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    return float((q3 + q1 - 2.0 * q2) / iqr)


def _safe_skew(x: np.ndarray, method: str = "auto") -> Tuple[float, str]:
    """
    Compute skewness, return (skew_value, method_used).
    - method="auto": if samples are near-constant or the moment method is unstable → fallback to Bowley
    - method="moment": always use the moment method
    - method="quantile": always use Bowley
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return (np.nan, "na")

    std = x.std()
    rng = x.max() - x.min()
    near_constant = (std == 0) or (rng == 0) or np.isclose(std, 0.0, rtol=0, atol=1e-12)

    if method == "moment":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = moment_skew(x, bias=True)
        return (float(v), "moment")

    if method == "quantile" or near_constant:
        return (_bowley_skew(x), "quantile")

    # auto: first try the moment method, fallback if numerically unstable
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        v = moment_skew(x, bias=True)
        unstable = any("Precision loss" in str(w.message) for w in wlist)
    if unstable or not np.isfinite(v):
        return (_bowley_skew(x), "quantile")
    return (float(v), "moment")


def calc_time_distribution_stats(
    df: pd.DataFrame,
    group_col: Optional[Union[str, List[str]]] = None,
    tmrca_col: str = "Time_years",
    ratio_quantile: float = 0.01,
    gmm_max_components: int = 5,
    gmm_min_samples: int = 10,
    random_state: int = 42,
    skew_method: str = "auto",
) -> pd.DataFrame:
    """
    Calculate metrics related to the temporal distribution structure
    
    Returns columns:
    - grouping columns
    - Count: sample size
    - Range: range on the log10 scale
    - StdDev: standard deviation on the log10 scale
    - Skewness: skewness on the log10 scale
    - Skew_method: method used to compute skewness
    - Estimated_Peaks: number of peaks estimated by GMM
    - Ratio(Max/Pq_nonzero): ratio of max to the positive quantile
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

        # Positive sample quantile (for ratio calculation)
        positives = vals[vals > 0]
        if len(positives) > 0:
            p_nonzero = float(np.quantile(positives, ratio_quantile))
            if p_nonzero <= 0:
                p_nonzero = np.nan
        else:
            p_nonzero = np.nan
        ratio_val = (v_max / p_nonzero) if (p_nonzero is not np.nan) and np.isfinite(p_nonzero) else np.nan

        # Statistics on the log10 scale
        log_std = float("nan")
        log_rng = float("nan")
        skew_val = float("nan")
        skew_used = "na"
        log_vals_np = None
        
        if len(positives) >= 2:
            log_vals_np = np.log10(positives.values)
            log_rng = float(log_vals_np.max() - log_vals_np.min())
            log_std = float(pd.Series(log_vals_np).std())
            skew_val, skew_used = _safe_skew(log_vals_np, method=skew_method)

        # If log_std is undefined (too few positive values), fall back to original values
        if math.isnan(log_std):
            log_std = float(vals.std()) if len(vals) >= 2 else float("nan")
        if math.isnan(log_rng):
            log_rng = float(v_max - v_min)

        # GMM peak estimation
        est_peaks = np.nan
        uniq = None
        if log_vals_np is not None:
            uniq = np.unique(log_vals_np)
        if log_vals_np is not None and (len(log_vals_np) >= gmm_min_samples) and (uniq.size >= 2):
            max_k = max(1, min(gmm_max_components, uniq.size))
            X = log_vals_np.reshape(-1, 1)
            cand = list(range(1, max_k + 1))
            bic_scores: List[float] = []
            for k in cand:
                try:
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type="full",
                        random_state=random_state
                    )
                    gmm.fit(X)
                    bic_scores.append(gmm.bic(X))
                except Exception:
                    bic_scores.append(np.inf)
            est_peaks = float(cand[int(np.argmin(bic_scores))]) if not all(np.isinf(b) for b in bic_scores) else np.nan

        # Build result row
        row: Dict[str, object] = {
            "Count": int(n),
            "Range": log_rng,
            "StdDev": log_std,
            "Skewness": float(skew_val),
            "Skew_method": skew_used,
            "Estimated_Peaks": est_peaks,
            f"Ratio(Max/P{int(ratio_quantile*100):02d}_nonzero)": ratio_val,
        }

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
    ratio_quantile: float = 0.01,
    gmm_max_components: int = 5,
    gmm_min_samples: int = 10,
    random_state: int = 42,
    skew_method: str = "auto",
    print_result: bool = True,
) -> pd.DataFrame:
    """
    Execute time distribution metric calculation.
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
    if gmm_max_components < 1:
        raise ValueError("--gmm_max_components must be >= 1")
    if gmm_min_samples < 0:
        raise ValueError("--gmm_min_samples must be >= 0")

    if isinstance(group_col, (list, tuple)):
        parsed_group_col = list(group_col)
    elif group_col is None:
        parsed_group_col = None
    else:
        parsed_group_col = _parse_group_col(group_col)  # type: ignore[arg-type]
    df = pd.read_csv(csv_path)

    res = calc_time_distribution_stats(
        df,
        group_col=parsed_group_col,
        tmrca_col=tmrca_col,
        ratio_quantile=float(ratio_quantile),
        gmm_max_components=int(gmm_max_components),
        gmm_min_samples=int(gmm_min_samples),
        random_state=int(random_state),
        skew_method=skew_method,
    )

    if print_result:
        if res.empty:
            print("Empty result (possibly too few samples or mismatched column names).")
        else:
            if have_tabulate:
                print(tabulate(res, headers="keys", tablefmt="github", showindex=False))
            else:
                print(res.to_string(index=False))

    output_csv = out_csv if out_csv is not None else csv_path.parent / "time_distribution_stats.csv"
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
        description="Calculate temporal distribution metrics (Range, StdDev, Skewness, Estimated_Peaks)"
    )
    parser.add_argument("--csv_path", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--out_csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--group_col", type=str, default="Continent",
                        help='Grouping columns: single column like "Continent", or multiple like "Continent,Country"; pass "none" for no grouping')
    parser.add_argument("--tmrca_col", type=str, default="Time_years", help="TMRCA value column name")
    parser.add_argument("--ratio_quantile", type=float, default=0.01, help="Positive-sample quantile (0-1) used as denominator")
    parser.add_argument("--gmm_max_components", type=int, default=5, help="Maximum number of GMM components (for peak estimation)")
    parser.add_argument("--gmm_min_samples", type=int, default=10, help="Minimum sample size threshold to enable GMM")
    parser.add_argument("--random_state", type=int, default=42, help="GMM random seed")
    parser.add_argument("--skew_method", type=str, default="auto", choices=["auto", "moment", "quantile"],
                        help="Skewness calculation method")

    args = parser.parse_args()

    # Validation
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv_path}")
    if not (0.0 < args.ratio_quantile < 1.0):
        raise ValueError(f"--ratio_quantile must be within (0,1): current {args.ratio_quantile}")
    if args.gmm_max_components < 1:
        raise ValueError("--gmm_max_components must be >= 1")
    if args.gmm_min_samples < 0:
        raise ValueError("--gmm_min_samples must be >= 0")

    run(
        csv_path=args.csv_path,
        out_csv=args.out_csv,
        group_col=args.group_col,
        tmrca_col=args.tmrca_col,
        ratio_quantile=float(args.ratio_quantile),
        gmm_max_components=int(args.gmm_max_components),
        gmm_min_samples=int(args.gmm_min_samples),
        random_state=int(args.random_state),
        skew_method=args.skew_method,
        print_result=True,
    )


if __name__ == "__main__":
    main()
