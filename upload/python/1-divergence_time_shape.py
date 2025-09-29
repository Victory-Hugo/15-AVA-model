#!/usr/bin/env python3
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict

import argparse
import os
import warnings


# Blue color formatting for print statements
def blue_print(message, *args, **kwargs):
    """Print text in blue color with white variables for better readability"""
    # ANSI color codes
    BLUE = '\033[34m'
    WHITE = '\033[37m' 
    RESET = '\033[0m'
    
    if args or kwargs:
        # Format message with white variables
        formatted_msg = message.format(*args, **kwargs)
        # Replace variable placeholders with white color
        import re
        # Find {} placeholders and color them white
        result = re.sub(r'(\{[^}]*\})', f'{WHITE}\1{BLUE}', formatted_msg)
        print(f"{BLUE}{result}{RESET}")
    else:
        print(f"{BLUE}{message}{RESET}")

def format_status(status, message, value=""):
    """Format status messages with consistent structure"""
    BLUE = '\033[34m'
    WHITE = '\033[37m'
    GREEN = '\033[32m' 
    RESET = '\033[0m'
    
    if value:
        print(f"{GREEN}[{status}]{RESET} {BLUE}{message}{RESET} {WHITE}{value}{RESET}")
    else:
        print(f"{GREEN}[{status}]{RESET} {BLUE}{message}{RESET}")

import numpy as np
import pandas as pd
from scipy.stats import skew as moment_skew
from sklearn.mixture import GaussianMixture


# ------------------------- Helper: Quantile Skewness & Robust Skewness -------------------------
def _bowley_skew(x: np.ndarray) -> float:
    """Bowley (quantile) skewness; returns 0.0 when IQR=0"""
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    return float((q3 + q1 - 2.0 * q2) / iqr)


def _safe_skew(x: np.ndarray, method: str = "auto") -> Tuple[float, str]:
    """
    Calculate skewness, returns (skew_value, method_used).
    - method="auto": If sample is approximately constant or moment method is unstable → fallback to Bowley
    - method="moment": Always use moment method
    - method="quantile": Always use Bowley
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

    # auto: try moment method first, fallback if numerically unstable
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        v = moment_skew(x, bias=True)
        unstable = any("Precision loss" in str(w.message) for w in wlist)
    if unstable or not np.isfinite(v):
        return (_bowley_skew(x), "quantile")
    return (float(v), "moment")


# ------------------------- Main Function: Output Two Types of Functional Indicators -------------------------
def calc_tmrca_stats(
    df: pd.DataFrame,
    group_col: Optional[Union[str, List[str]]] = None,
    tmrca_col: str = "Time_years",
    ancient_threshold: float = 100000.0,     # AncientRatio 阈值
    ratio_quantile: float = 0.01,            # 正值样本分位数，做稳健分母
    gmm_max_components: int = 5,
    gmm_min_samples: int = 10,
    random_state: int = 42,
    skew_method: str = "auto"                # "auto" | "moment" | "quantile"
) -> pd.DataFrame:
    """
    Group df by (single or multiple columns), calculate indicators of two types of functions and return DataFrame.
    Only depends on `tmrca_col`, no normalization, weighting or labeling will be performed.
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

        # ----------- Basic Statistics -----------
        v_max = float(vals.max())
        v_min = float(vals.min())
        v_rng = float(v_max - v_min)
        v_std = float(vals.std())  # pandas default ddof=1 (sample standard deviation)

        # Positive sample quantile (robust denominator)
        positives = vals[vals > 0]
        if len(positives) > 0:
            p_nonzero = float(np.quantile(positives, ratio_quantile))
            if p_nonzero <= 0:
                p_nonzero = np.nan
        else:
            p_nonzero = np.nan
        ratio_val = (v_max / p_nonzero) if (p_nonzero is not np.nan) and np.isfinite(p_nonzero) else np.nan

        # Skewness (robust)
        skew_val, skew_used = _safe_skew(vals.values, method=skew_method)

        # Ancient lineage ratio
        ancient_ratio = float((vals > ancient_threshold).mean())

        # ----------- Peak Number Estimation (GMM + BIC) -----------
        uniq = np.unique(vals.values)
        if (n >= gmm_min_samples) and (uniq.size >= 2):
            max_k = max(1, min(gmm_max_components, uniq.size))
            X = vals.values.reshape(-1, 1)
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
        else:
            est_peaks = np.nan

        # Expand group name to columns
        row: Dict[str, object] = {
            "Count": int(n),
            "Max": v_max,
            "Min": v_min,
            "Min_nonzero": float(positives.min()) if len(positives) > 0 else np.nan,
            "Pq_nonzero": p_nonzero,  # q = ratio_quantile
            "Range": v_rng,
            f"Ratio(Max/P{int(ratio_quantile*100):02d}_nonzero)": ratio_val,
            "StdDev": v_std,
            "Skewness": float(skew_val),
            "Skew_method": skew_used,
            f"AncientRatio(>{int(ancient_threshold)})": ancient_ratio,
            "Estimated_Peaks": est_peaks,
        }

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

    # Column order: grouping columns first, others in order of indicator appearance
    front_cols = (group_cols or ["Group"])
    metric_cols = [c for c in out.columns if c not in front_cols]
    return out[front_cols + metric_cols]


# ------------------------- CLI Entry -------------------------
def _parse_group_col(arg: str) -> Optional[Union[str, List[str]]]:
    """
    Parse command line group_col:
    - "none" / "null" / "" → None (no grouping)
    - "A" → "A"
    - "A,B,C" → ["A", "B", "C"]
    """
    if arg is None:
        return "Continent"  # default
    s = arg.strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip() != ""]
    return s


def main():
    try:
        from tabulate import tabulate
        have_tabulate = True
    except Exception:
        have_tabulate = False

    parser = argparse.ArgumentParser(
        description="Calculate TMRCA statistics by group and output CSV (core logic consistent with original script, only parameters changed to command line input)"
    )
    parser.add_argument("--csv_path", required=True, type=Path, help="Input CSV path (required)")
    parser.add_argument("--out_csv", type=Path, default=None, help="Output CSV path (default: tmrca_stats.csv in same directory as input)")

    parser.add_argument("--group_col", type=str, default="Continent",
                        help='Grouping column: single column like "Continent", or multiple columns like "Continent,Country"; pass "none" for no grouping')
    parser.add_argument("--tmrca_col", type=str, default="Time_years", help="TMRCA numeric column name")
    parser.add_argument("--ancient_threshold", type=float, default=100000.0, help="AncientRatio threshold")
    parser.add_argument("--ratio_quantile", type=float, default=0.01, help="Quantile of positive samples for denominator (0-1)")
    parser.add_argument("--gmm_max_components", type=int, default=5, help="GMM maximum number of components (for peak number estimation)")
    parser.add_argument("--gmm_min_samples", type=int, default=10, help="Minimum sample size threshold for enabling GMM")
    parser.add_argument("--random_state", type=int, default=42, help="GMM random seed")
    parser.add_argument("--skew_method", type=str, default="auto", choices=["auto", "moment", "quantile"],
                        help="Skewness calculation method")

    args = parser.parse_args()

    # Validation and preprocessing
    if not args.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv_path}")

    if not (0.0 < args.ratio_quantile < 1.0):
        raise ValueError(f"--ratio_quantile must be in (0,1): current {args.ratio_quantile}")

    if args.gmm_max_components < 1:
        raise ValueError("--gmm_max_components must be >= 1")
    if args.gmm_min_samples < 0:
        raise ValueError("--gmm_min_samples must be >= 0")

    group_col = _parse_group_col(args.group_col)

    # Read input
    df = pd.read_csv(args.csv_path)

    # Run calculation
    res = calc_tmrca_stats(
        df,
        group_col=group_col,
        tmrca_col=args.tmrca_col,
        ancient_threshold=float(args.ancient_threshold),
        ratio_quantile=float(args.ratio_quantile),
        gmm_max_components=int(args.gmm_max_components),
        gmm_min_samples=int(args.gmm_min_samples),
        random_state=int(args.random_state),
        skew_method=args.skew_method,
    )

    # Print results
    if res.empty:
        blue_print("Result is empty (possibly too few samples or column name mismatch).")
    else:
        if have_tabulate:
            print(tabulate(res, headers="keys", tablefmt="github", showindex=False))
        else:
            # Fallback printing
            print(res.to_string(index=False))

    # Write output
    out_csv: Path
    if args.out_csv is not None:
        out_csv = args.out_csv
    else:
        out_csv = args.csv_path.parent / "tmrca_stats.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    format_status("SAVED", "Results saved to:", str(out_csv.resolve()))


if __name__ == "__main__":
    main()
