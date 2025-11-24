#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


处理逻辑：
1) 合并：按 Continent 外连接（outer merge），不做重命名。
2) PCA+归一化：
   - 时间深度轴：将 Max 与 AncientRatio 做 z-score 后 PCA 得第一主成分并方向校准，再映射到 [0,1]
   - 时间结构轴：将 StdDev、Range、Skewness、Estimated_Peaks 做 z-score 后 PCA 得第一主成分并方向校准，再映射到 [0,1]
   - 其余 Diversity_pattern_score、Unique_hap_score 直接 z-score→Φ(z)
3) 加权求和：对“归一化后的轴/得分”乘以权重相加得到 All_score_raw
4) 总分归一化：All_score = Φ( z(All_score_raw) )
"""

import re
import math
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def zscore_to_cdf01(x: pd.Series) -> pd.Series:
    """将一列经 z-score 后映射到 [0,1]：Φ(z)"""
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        # 常量列：全部给 0.5（Φ(0)）
        return pd.Series(np.full(len(x), 0.5), index=x.index)
    z = (x - mu) / sigma
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2)))


def minmax_normalize(x: pd.Series) -> pd.Series:
    """将一列归一化到 0-1 范围"""
    x_min = x.min()
    x_max = x.max()
    if x_min == x_max or np.isnan(x_min) or np.isnan(x_max):
        return pd.Series(np.full(len(x), 0.5), index=x.index)
    return (x - x_min) / (x_max - x_min)


def find_ancient_col_and_threshold(columns) -> tuple[str | None, float | None]:
    """
    在列名中寻找 AncientRatio(>XXXXX) 模式，解析阈值 T。
    返回 (列名, 阈值)；若未找到，返回 (None, None)
    """
    for c in columns:
        if isinstance(c, str) and c.startswith("AncientRatio("):
            m = re.search(r"AncientRatio\(\>([0-9]+(?:\.[0-9]*)?)\)", c)
            if m:
                return c, float(m.group(1))
            else:
                # 尝试更宽松的解析
                m2 = re.search(r"\>([0-9]+(?:\.[0-9]*)?)", c)
                if m2:
                    return c, float(m2.group(1))
                return c, None
    return None, None


def align_pca_direction(component: pd.Series, reference: pd.Series) -> pd.Series:
    """
    若 PCA 得到的主成分与参考指标负相关，则整体取反，确保“参考指标越大 → 主成分越大”。
    若有效样本不足或相关系数为 NaN，则保持原状。
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


def main():
    ap = argparse.ArgumentParser(description="Merge, normalize, weight, score, and classify population metrics.")
    ap.add_argument("--tmrca", required=True, help="Absolute path to Final_tmrca_stats.csv")
    ap.add_argument("--amova", required=True, help="Absolute path to Final_AMOVA_scores.csv")
    ap.add_argument("--unique", required=True, help="Absolute path to Final_unique_hap.csv")
    ap.add_argument("--out",   required=True, help="Absolute path to output scored CSV")
    ap.add_argument("--group-col", required=True, help="Column name for grouping (e.g., 'River', 'Continent')")

    # Weights (defaults per your spec)
    ap.add_argument("--w-time-depth", type=float, default=4.0, help="权重：时间深度轴（基于核得分或Max+AncientRatio PCA）")
    ap.add_argument("--w-time-structure", type=float, default=4.0, help="权重：时间结构轴（StdDev+Range+Skewness+Peaks PCA）")
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
    
    # 标准化分组字段的值（统一大小写）
    group_col = args.group_col
    if group_col in tmrca.columns:
        tmrca[group_col] = tmrca[group_col].str.strip().str.title()
    if group_col in amova.columns:
        amova[group_col] = amova[group_col].str.strip().str.title()
    if group_col in uhap.columns:
        uhap[group_col] = uhap[group_col].str.strip().str.title()

    # --- Basic column existence checks ---
    group_col = args.group_col
    if group_col not in tmrca.columns:
        raise ValueError(f"TMRCA file must contain a '{group_col}' column.")
    if group_col not in amova.columns:
        raise ValueError(f"AMOVA file must contain a '{group_col}' column.")
    if group_col not in uhap.columns:
        raise ValueError(f"Unique hap file must contain a '{group_col}' column.")

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

    # --- Merge (outer join on group column) ---
    merged = tmrca.merge(amova, on=group_col, how="outer").merge(uhap, on=group_col, how="outer")

    # --- Detect AncientRatio column & threshold for Max rule ---
    ancient_col, ancient_thr = find_ancient_col_and_threshold(merged.columns)

    # --- 时间深度轴：优先使用 TimeDepthKernel，否则基于核得分或回退到 Max+AncientRatio PCA
    td_kernel_cols = [c for c in merged.columns if isinstance(c, str) and c.startswith("TimeDepthKernel")]
    if td_kernel_cols:
        # 方案1：直接使用预计算的TimeDepthKernel
        td_col = td_kernel_cols[0]
        merged["time_depth_axis"] = merged[td_col]
        merged["time_depth_axis_norm"] = zscore_to_cdf01(merged["time_depth_axis"])
    else:
        # 方案2：基于多个核得分进行PCA
        kernel_cols = [c for c in merged.columns if isinstance(c, str) and c.startswith("KernelScore_T")]
        if kernel_cols:
            # 对核得分进行 z-score 预处理后PCA
            kernel_df = merged[kernel_cols].copy()
            
            # 处理NaN值
            for col in kernel_df.columns:
                if kernel_df[col].isna().any():
                    kernel_df[col] = kernel_df[col].fillna(kernel_df[col].mean())
            
            # z-score标准化
            for col in kernel_df.columns:
                mu = kernel_df[col].mean()
                sigma = kernel_df[col].std(ddof=0)
                if sigma != 0 and not np.isnan(sigma):
                    kernel_df[col] = (kernel_df[col] - mu) / sigma
            
            # 填充剩余NaN
            if kernel_df.isna().any().any():
                kernel_df = kernel_df.fillna(0)
            
            # PCA提取第一主成分
            pca_kernel = PCA(n_components=1, random_state=42)
            kernel_pca = pca_kernel.fit_transform(kernel_df)
            merged["time_depth_axis"] = kernel_pca[:, 0]
            
            # 方向校准：确保与第一个核得分正相关
            ref_col = kernel_cols[0] if kernel_cols else None
            if ref_col:
                merged["time_depth_axis"] = align_pca_direction(merged["time_depth_axis"], merged[ref_col])
            
            merged["time_depth_axis_norm"] = zscore_to_cdf01(merged["time_depth_axis"])
        else:
            # 方案3：回退到传统的Max+AncientRatio PCA
            time_depth_vars = []
            if "Max" in merged.columns:
                time_depth_vars.append("Max")
            if ancient_col is not None:
                time_depth_vars.append(ancient_col)

            if time_depth_vars:
                time_depth_df = merged[time_depth_vars].copy()
                
                # 处理NaN值：用列的均值填充
                for col in time_depth_df.columns:
                    if time_depth_df[col].isna().any():
                        time_depth_df[col] = time_depth_df[col].fillna(time_depth_df[col].mean())
                
                for col in time_depth_df.columns:
                    mu = time_depth_df[col].mean()
                    sigma = time_depth_df[col].std(ddof=0)
                    if sigma != 0 and not np.isnan(sigma):
                        time_depth_df[col] = (time_depth_df[col] - mu) / sigma

                # 检查是否仍有NaN值，如果有则用0填充
                if time_depth_df.isna().any().any():
                    time_depth_df = time_depth_df.fillna(0)

                pca_time_depth = PCA(n_components=1, random_state=42)
                time_depth_pca = pca_time_depth.fit_transform(time_depth_df)
                merged["time_depth_axis"] = time_depth_pca[:, 0]
                ref_col = "Max" if "Max" in merged.columns else time_depth_vars[0]
                merged["time_depth_axis"] = align_pca_direction(merged["time_depth_axis"], merged[ref_col])
                merged["time_depth_axis_norm"] = zscore_to_cdf01(merged["time_depth_axis"])

    # 2. 时间结构复杂度轴 (Time Structure Complexity Axis): StdDev + Range + Skewness + Estimated_Peaks
    time_struct_vars = ["StdDev", "Range", "Skewness", "Estimated_Peaks"]
    available_struct_vars = [var for var in time_struct_vars if var in merged.columns]

    if available_struct_vars:
        # 对时间结构变量进行 z-score 预处理
        time_struct_df = merged[available_struct_vars].copy()
        
        # 处理NaN值：用列的均值填充
        for col in time_struct_df.columns:
            if time_struct_df[col].isna().any():
                time_struct_df[col] = time_struct_df[col].fillna(time_struct_df[col].mean())
        
        for col in time_struct_df.columns:
            mu = time_struct_df[col].mean()
            sigma = time_struct_df[col].std(ddof=0)
            if sigma != 0 and not np.isnan(sigma):
                time_struct_df[col] = (time_struct_df[col] - mu) / sigma

        # 检查是否仍有NaN值，如果有则用0填充
        if time_struct_df.isna().any().any():
            time_struct_df = time_struct_df.fillna(0)

        # 进行 PCA 并提取第一主成分
        pca_time_struct = PCA(n_components=1, random_state=42)
        time_struct_pca = pca_time_struct.fit_transform(time_struct_df)
        merged["time_structure_axis"] = time_struct_pca[:, 0]
        ref_col = "StdDev" if "StdDev" in merged.columns else available_struct_vars[0]
        merged["time_structure_axis"] = align_pca_direction(merged["time_structure_axis"], merged[ref_col])

        # 使用 z-score→Φ 归一化到 0-1 范围
        merged["time_structure_axis_norm"] = zscore_to_cdf01(merged["time_structure_axis"])

    # 3. 其他原有归一化指标 (保持不变)
    merged["Diversity_pattern_score_norm"] = zscore_to_cdf01(merged["Diversity_pattern_score"])
    merged["Unique_hap_score_norm"]       = zscore_to_cdf01(merged["Unique_hap_score"])

    # 注意：核得分已经被整合到时间深度轴中，不再单独作为评分维度

    # --- Build weighted sum on normalized columns ---
    # 使用四个主要维度进行加权求和（核得分已整合到时间深度轴中）
    weights = {
        "time_depth_axis_norm": args.w_time_depth,
        "time_structure_axis_norm": args.w_time_structure,
        "Diversity_pattern_score_norm": args.w_diversity,
        "Unique_hap_score_norm": args.w_unique,
    }

    # 仅对存在于数据中的列参与加权
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

    # --- Save only selected columns ---
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

    # Verify all required columns exist
    missing_columns = [col for col in required_columns if col not in merged.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Write only the required columns
    merged[required_columns].to_csv(args.out, index=False)

    # --- Console summary ---
    print(f"[OK] Wrote: {args.out}")
    print("Used weights on columns present:")
    for c in use_cols:
        print(f"  - {c}: w={weights[c]}")
    if ancient_col is not None:
        print(f"Ancient column: {ancient_col}")
    if ancient_thr is not None:
        print(f"Max binary threshold T from AncientRatio: T = {ancient_thr:g}")
    print(f"Classification thresholds: Origin_like if > {thr_origin}, Mix_like if ({thr_mix_low}, {thr_origin}], else Middle_like")


if __name__ == "__main__":
    main()
