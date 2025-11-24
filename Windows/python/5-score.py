#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Processing logic:
1) Merge: outer join on Continent (no renaming).
2) PCA + normalization:
   - Time depth axis: z-score Max and AncientRatio, run PCA to get the first component, align direction, then map to [0,1]
   - Time structure axis: z-score StdDev, Range, Skewness, Estimated_Peaks, run PCA to get the first component, align direction, then map to [0,1]
   - Other metrics Diversity_pattern_score, Unique_hap_score go through z-score→Φ(z) directly
3) Weighted sum: multiply normalized axes/scores by weights and sum to get All_score_raw
4) Normalize total score: All_score = Φ( z(All_score_raw) )
"""

import re
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def zscore_to_cdf01(x: pd.Series) -> pd.Series:
    """Map a column to [0,1] via z-score then Φ(z)"""
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        # Constant column: give all 0.5 (Φ(0))
        return pd.Series(np.full(len(x), 0.5), index=x.index)
    z = (x - mu) / sigma
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2)))


def minmax_normalize(x: pd.Series) -> pd.Series:
    """Normalize a column to the 0-1 range"""
    x_min = x.min()
    x_max = x.max()
    if x_min == x_max or np.isnan(x_min) or np.isnan(x_max):
        return pd.Series(np.full(len(x), 0.5), index=x.index)
    return (x - x_min) / (x_max - x_min)


def find_ancient_col_and_threshold(columns) -> tuple[str | None, float | None]:
    """
    Look for AncientRatio(>XXXXX) pattern in column names and parse threshold T.
    Return (column_name, threshold); if not found, return (None, None)
    """
    for c in columns:
        if isinstance(c, str) and c.startswith("AncientRatio("):
            m = re.search(r"AncientRatio\(\>([0-9]+(?:\.[0-9]*)?)\)", c)
            if m:
                return c, float(m.group(1))
            else:
                # Try a looser parse
                m2 = re.search(r"\>([0-9]+(?:\.[0-9]*)?)", c)
                if m2:
                    return c, float(m2.group(1))
                return c, None
    return None, None


def align_pca_direction(component: pd.Series, reference: pd.Series) -> pd.Series:
    """
    If the PCA component is negatively correlated with the reference metric, flip the sign to ensure "higher reference → higher component".
    If there are insufficient valid samples or correlation is NaN, keep as-is.
    """
    if component is None or reference is None:
        return component
    mask = component.notna() & reference.notna()
    if mask.sum() < 2:
        return component
    corr = np.corrcoef(component[mask], reference[mask])[0, 1]
    if np.isnan(corr) or corr >= 0:
        return component
    return -component


def run(
    tmrca_path: str,
    amova_path: str,
    unique_path: str,
    out_path: str,
    group_col: str,
    w_time_depth: float = 4.0,
    w_time_structure: float = 4.0,
    w_diversity: float = 1.5,
    w_unique: float = 1.5,
    thr_origin: float = 0.6,
    thr_mix_low: float = 0.4,
    print_result: bool = True,
) -> pd.DataFrame:
    """
    Merge, normalize, weight, score, and classify population metrics.
    """
    tmrca = pd.read_csv(tmrca_path)
    amova = pd.read_csv(amova_path)
    uhap = pd.read_csv(unique_path)

    if group_col in tmrca.columns:
        tmrca[group_col] = tmrca[group_col].str.strip().str.title()
    if group_col in amova.columns:
        amova[group_col] = amova[group_col].str.strip().str.title()
    if group_col in uhap.columns:
        uhap[group_col] = uhap[group_col].str.strip().str.title()

    if group_col not in tmrca.columns:
        raise ValueError(f"TMRCA file must contain a '{group_col}' column.")
    if group_col not in amova.columns:
        raise ValueError(f"AMOVA file must contain a '{group_col}' column.")
    if group_col not in uhap.columns:
        raise ValueError(f"Unique hap file must contain a '{group_col}' column.")

    required_tmrca_cols = ["Max", "StdDev", "Range", "Skewness", "Estimated_Peaks"]
    missing_tmrca = [c for c in required_tmrca_cols if c not in tmrca.columns]
    if missing_tmrca:
        raise ValueError(f"TMRCA file missing required columns: {missing_tmrca}")

    if "Diversity_pattern_score" not in amova.columns:
        raise ValueError("AMOVA file must contain 'Diversity_pattern_score'.")
    if "Unique_hap_score" not in uhap.columns:
        raise ValueError("Unique hap file must contain 'Unique_hap_score'.")

    merged = tmrca.merge(amova, on=group_col, how="outer").merge(uhap, on=group_col, how="outer")

    ancient_col, ancient_thr = find_ancient_col_and_threshold(merged.columns)

    td_kernel_cols = [c for c in merged.columns if isinstance(c, str) and c.startswith("TimeDepthKernel")]
    if td_kernel_cols:
        td_col = td_kernel_cols[0]
        merged["time_depth_axis"] = merged[td_col]
        merged["time_depth_axis_norm"] = zscore_to_cdf01(merged["time_depth_axis"])
    else:
        kernel_cols = [c for c in merged.columns if isinstance(c, str) and c.startswith("KernelScore_T")]
        if kernel_cols:
            kernel_df = merged[kernel_cols].copy()

            for col in kernel_df.columns:
                if kernel_df[col].isna().any():
                    kernel_df[col] = kernel_df[col].fillna(kernel_df[col].mean())

            for col in kernel_df.columns:
                mu = kernel_df[col].mean()
                sigma = kernel_df[col].std(ddof=0)
                if sigma != 0 and not np.isnan(sigma):
                    kernel_df[col] = (kernel_df[col] - mu) / sigma

            if kernel_df.isna().any().any():
                kernel_df = kernel_df.fillna(0)

            pca_kernel = PCA(n_components=1, random_state=42)
            kernel_pca = pca_kernel.fit_transform(kernel_df)
            merged["time_depth_axis"] = kernel_pca[:, 0]

            ref_col = kernel_cols[0] if kernel_cols else None
            if ref_col:
                merged["time_depth_axis"] = align_pca_direction(merged["time_depth_axis"], merged[ref_col])

            merged["time_depth_axis_norm"] = zscore_to_cdf01(merged["time_depth_axis"])
        else:
            time_depth_vars = []
            if "Max" in merged.columns:
                time_depth_vars.append("Max")
            if ancient_col is not None:
                time_depth_vars.append(ancient_col)

            if time_depth_vars:
                time_depth_df = merged[time_depth_vars].copy()

                for col in time_depth_df.columns:
                    if time_depth_df[col].isna().any():
                        time_depth_df[col] = time_depth_df[col].fillna(time_depth_df[col].mean())

                for col in time_depth_df.columns:
                    mu = time_depth_df[col].mean()
                    sigma = time_depth_df[col].std(ddof=0)
                    if sigma != 0 and not np.isnan(sigma):
                        time_depth_df[col] = (time_depth_df[col] - mu) / sigma

                if time_depth_df.isna().any().any():
                    time_depth_df = time_depth_df.fillna(0)

                pca_time_depth = PCA(n_components=1, random_state=42)
                time_depth_pca = pca_time_depth.fit_transform(time_depth_df)
                merged["time_depth_axis"] = time_depth_pca[:, 0]
                ref_col = "Max" if "Max" in merged.columns else time_depth_vars[0]
                merged["time_depth_axis"] = align_pca_direction(merged["time_depth_axis"], merged[ref_col])
                merged["time_depth_axis_norm"] = zscore_to_cdf01(merged["time_depth_axis"])

    time_struct_vars = ["StdDev", "Range", "Skewness", "Estimated_Peaks"]
    available_struct_vars = [var for var in time_struct_vars if var in merged.columns]

    if available_struct_vars:
        time_struct_df = merged[available_struct_vars].copy()

        for col in time_struct_df.columns:
            if time_struct_df[col].isna().any():
                time_struct_df[col] = time_struct_df[col].fillna(time_struct_df[col].mean())

        for col in time_struct_df.columns:
            mu = time_struct_df[col].mean()
            sigma = time_struct_df[col].std(ddof=0)
            if sigma != 0 and not np.isnan(sigma):
                time_struct_df[col] = (time_struct_df[col] - mu) / sigma

        if time_struct_df.isna().any().any():
            time_struct_df = time_struct_df.fillna(0)

        pca_time_struct = PCA(n_components=1, random_state=42)
        time_struct_pca = pca_time_struct.fit_transform(time_struct_df)
        merged["time_structure_axis"] = time_struct_pca[:, 0]
        ref_col = "StdDev" if "StdDev" in merged.columns else available_struct_vars[0]
        merged["time_structure_axis"] = align_pca_direction(merged["time_structure_axis"], merged[ref_col])

        merged["time_structure_axis_norm"] = zscore_to_cdf01(merged["time_structure_axis"])

    merged["Diversity_pattern_score_norm"] = zscore_to_cdf01(merged["Diversity_pattern_score"])
    merged["Unique_hap_score_norm"] = zscore_to_cdf01(merged["Unique_hap_score"])

    weights = {
        "time_depth_axis_norm": w_time_depth,
        "time_structure_axis_norm": w_time_structure,
        "Diversity_pattern_score_norm": w_diversity,
        "Unique_hap_score_norm": w_unique,
    }

    use_cols = [c for c in weights.keys() if c in merged.columns]
    if not use_cols:
        raise ValueError("No normalized columns available for scoring.")

    all_score_raw = None
    for c in use_cols:
        term = merged[c] * weights[c]
        all_score_raw = term if all_score_raw is None else (all_score_raw + term)
    merged["All_score_raw"] = all_score_raw

    thr_origin = float(thr_origin)
    thr_mix_low = float(thr_mix_low)
    if not (0.0 <= thr_mix_low < thr_origin <= 1.0):
        raise ValueError("Require 0 ≤ thr_mix_low < thr_origin ≤ 1")

    merged["All_score"] = zscore_to_cdf01(merged["All_score_raw"])

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

    required_columns = [
        group_col,
        "Count",
        "time_depth_axis",
        "time_depth_axis_norm",
        "time_structure_axis",
        "time_structure_axis_norm",
        "Diversity_pattern_score_norm",
        "Unique_hap_score_norm",
        "All_score_raw",
        "All_score",
        "Class_label"
    ]

    missing_columns = [col for col in required_columns if col not in merged.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged[required_columns].to_csv(out_path, index=False)

    if print_result:
        print(f"[OK] Wrote: {out_path}")
        print("Used weights on columns present:")
        for c in use_cols:
            print(f"  - {c}: w={weights[c]}")
        if ancient_col is not None:
            print(f"Ancient column: {ancient_col}")
        if ancient_thr is not None:
            print(f"Max binary threshold T from AncientRatio: T = {ancient_thr:g}")
        print(f"Classification thresholds: Origin_like if > {thr_origin}, Mix_like if ({thr_mix_low}, {thr_origin}], else Middle_like")
    return merged[required_columns]


def main():
    ap = argparse.ArgumentParser(description="Merge, normalize, weight, score, and classify population metrics.")
    ap.add_argument("--tmrca", required=True, help="Absolute path to Final_tmrca_stats.csv")
    ap.add_argument("--amova", required=True, help="Absolute path to Final_AMOVA_scores.csv")
    ap.add_argument("--unique", required=True, help="Absolute path to Final_unique_hap.csv")
    ap.add_argument("--out", required=True, help="Absolute path to output scored CSV")
    ap.add_argument("--group-col", required=True, help="Column name for grouping (e.g., 'River', 'Continent')")

    ap.add_argument("--w-time-depth", type=float, default=4.0, help="Weight: time depth axis (kernel scores or Max+AncientRatio PCA)")
    ap.add_argument("--w-time-structure", type=float, default=4.0, help="Weight: time structure axis (StdDev+Range+Skewness+Peaks PCA)")
    ap.add_argument("--w-diversity", type=float, default=1.5)
    ap.add_argument("--w-unique", type=float, default=1.5)

    ap.add_argument("--thr-origin", type=float, default=0.6, help="Origin_like if All_score > thr-origin")
    ap.add_argument("--thr-mix-low", type=float, default=0.4, help="Mix_like if thr-mix-low < All_score ≤ thr-origin")

    args = ap.parse_args()

    run(
        tmrca_path=args.tmrca,
        amova_path=args.amova,
        unique_path=args.unique,
        out_path=args.out,
        group_col=args.group_col,
        w_time_depth=float(args.w_time_depth),
        w_time_structure=float(args.w_time_structure),
        w_diversity=float(args.w_diversity),
        w_unique=float(args.w_unique),
        thr_origin=float(args.thr_origin),
        thr_mix_low=float(args.thr_mix_low),
        print_result=True,
    )


if __name__ == "__main__":
    main()
