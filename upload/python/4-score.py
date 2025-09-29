#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import math
import argparse
import numpy as np
import pandas as pd


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


def zscore_to_cdf01(x: pd.Series) -> pd.Series:
    """Apply z-score to a column then map to [0,1]: Φ(z)"""
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        # Constant column: all get 0.5 (Φ(0))
        return pd.Series(np.full(len(x), 0.5), index=x.index)
    z = (x - mu) / sigma
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2)))


def find_ancient_col_and_threshold(columns) -> tuple[str | None, float | None]:
    """
    Look for AncientRatio(>XXXXX) pattern in column names, parse threshold T.
    Returns (column_name, threshold); if not found, returns (None, None)
    """
    for c in columns:
        if isinstance(c, str) and c.startswith("AncientRatio("):
            m = re.search(r"AncientRatio\(\>([0-9]+(?:\.[0-9]*)?)\)", c)
            if m:
                return c, float(m.group(1))
            else:
                # Try more lenient parsing
                m2 = re.search(r"\>([0-9]+(?:\.[0-9]*)?)", c)
                if m2:
                    return c, float(m2.group(1))
                return c, None
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Merge, normalize, weight, score, and classify population metrics.")
    ap.add_argument("--tmrca", required=True, help="Absolute path to Final_tmrca_stats.csv")
    ap.add_argument("--amova", required=True, help="Absolute path to Final_AMOVA_scores.csv")
    ap.add_argument("--unique", required=True, help="Absolute path to Final_unique_hap.csv")
    ap.add_argument("--out",   required=True, help="Absolute path to output scored CSV")

    # Weights (defaults per your spec)
    ap.add_argument("--w-max", type=float, default=2.0)
    ap.add_argument("--w-ancient", type=float, default=2.0)
    ap.add_argument("--w-std", type=float, default=1.0)
    ap.add_argument("--w-range", type=float, default=1.0)
    ap.add_argument("--w-skew", type=float, default=1.0)
    ap.add_argument("--w-peaks", type=float, default=1.0)
    ap.add_argument("--w-diversity", type=float, default=1.5)
    ap.add_argument("--w-unique", type=float, default=1.5)

    # Classification thresholds
    ap.add_argument("--thr-origin", type=float, default=0.6, help="Origin_like if All_score > thr-origin")
    ap.add_argument("--thr-mix-low", type=float, default=0.4, help="Mix_like if thr-mix-low < All_score ≤ thr-origin")

    args = ap.parse_args()

    # --- Load data ---
    tmrca = pd.read_csv(args.tmrca)
    amova = pd.read_csv(args.amova)
    uhap  = pd.read_csv(args.unique)

    # --- Basic column existence checks ---
    if "Continent" not in tmrca.columns:
        raise ValueError("TMRCA file must contain a 'Continent' column.")
    if "Continent" not in amova.columns:
        raise ValueError("AMOVA file must contain a 'Continent' column.")
    if "Continent" not in uhap.columns:
        raise ValueError("Unique hap file must contain a 'Continent' column.")

    # TMRCA columns needed
    required_tmrca_cols = ["Max", "StdDev", "Range", "Skewness", "Estimated_Peaks"]
    missing_tmrca = [c for c in required_tmrca_cols if c not in tmrca.columns]
    if missing_tmrca:
        raise ValueError(f"TMRCA file missing required columns: {missing_tmrca}")

    # AMOVA & Unique columns needed
    if "Diversity_pattern_score" not in amova.columns:
        raise ValueError("AMOVA file must contain 'Diversity_pattern_score'.")
    if "Unique_hap_score" not in uhap.columns:
        raise ValueError("Unique hap file must contain 'Unique_hap_score'.")

    # --- Merge (outer join on Continent) ---
    merged = tmrca.merge(amova, on="Continent", how="outer").merge(uhap, on="Continent", how="outer")

    # --- Detect AncientRatio column & threshold for Max rule ---
    ancient_col, ancient_thr = find_ancient_col_and_threshold(merged.columns)

    # --- Per-column normalization ---
    # Max_norm: special binary rule if threshold found; else fallback to z-score→Φ
    if ancient_thr is not None and "Max" in merged.columns:
        merged["Max_norm"] = (merged["Max"] > ancient_thr).astype(float)
    elif "Max" in merged.columns:
        merged["Max_norm"] = zscore_to_cdf01(merged["Max"])

    # AncientRatio column normalization (if present)
    if ancient_col is not None:
        merged[f"{ancient_col}_norm"] = zscore_to_cdf01(merged[ancient_col])

    # Other TMRCA columns
    for col in ["StdDev", "Range", "Skewness", "Estimated_Peaks"]:
        if col in merged.columns:
            merged[f"{col}_norm"] = zscore_to_cdf01(merged[col])

    # AMOVA / Unique: also do z-score→Φ normalization first
    merged["Diversity_pattern_score_norm"] = zscore_to_cdf01(merged["Diversity_pattern_score"])
    merged["Unique_hap_score_norm"]       = zscore_to_cdf01(merged["Unique_hap_score"])

    # --- Build weighted sum on normalized columns ---
    # Map weights to the normalized feature names we actually use
    weights = {
        "Max_norm": args.w_max,
        f"{ancient_col}_norm": args.w_ancient if ancient_col is not None else 0.0,
        "StdDev_norm": args.w_std,
        "Range_norm": args.w_range,
        "Skewness_norm": args.w_skew,
        "Estimated_Peaks_norm": args.w_peaks,
        "Diversity_pattern_score_norm": args.w_diversity,
        "Unique_hap_score_norm": args.w_unique,
    }

    # Only include columns that exist in the data for weighting
    use_cols = [c for c in weights.keys() if c in merged.columns]
    if not use_cols:
        raise ValueError("No normalized columns available for scoring.")

    all_score_raw = None
    for c in use_cols:
        term = merged[c] * weights[c]
        all_score_raw = term if all_score_raw is None else (all_score_raw + term)
    merged["All_score_raw"] = all_score_raw

    # --- Normalize overall score to [0,1] via z-score→Φ ---
    merged["All_score"] = zscore_to_cdf01(merged["All_score_raw"])

    # --- Classification ---
    thr_origin = float(args.thr_origin)
    thr_mix_low = float(args.thr_mix_low)
    if not (0.0 <= thr_mix_low < thr_origin <= 1.0):
        raise ValueError("Require 0 ≤ thr_mix_low < thr_origin ≤ 1")

    def label_from_score(s):
        if pd.isna(s):
            return np.nan
        if s > thr_origin:
            return "Origin_like"
        elif s > thr_mix_low:
            return "Middle_like"
        else:
            return "Mix_like"

    merged["Class_label"] = merged["All_score"].apply(label_from_score)

    # --- Save ---
    merged.to_csv(args.out, index=False)

    # --- Console summary ---
    format_status("SAVED", "Final scored metrics:", args.out)
    blue_print("\n=== CONFIGURATION SUMMARY ===")
    blue_print("Weights applied to normalized columns:")
    for c in use_cols:
        blue_print(f"  • {c}: w={weights[c]}")
    
    if ancient_col is not None:
        blue_print(f"\nAncient lineage column: {ancient_col}")
    if ancient_thr is not None:
        blue_print(f"Max binary threshold from AncientRatio: T = {ancient_thr:g}")
    
    blue_print(f"\nClassification thresholds:")
    blue_print(f"  • Origin_like:  score > {thr_origin}")
    blue_print(f"  • Middle_like:  {thr_mix_low} < score ≤ {thr_origin}")
    blue_print(f"  • Mix_like:     score ≤ {thr_mix_low}")


if __name__ == "__main__":
    main()
