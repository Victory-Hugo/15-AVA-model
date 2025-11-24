#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间分布指标计算模块
功能：计算与时间分布结构相关的指标
- Range (log10 尺度)
- StdDev (log10 尺度)
- Skewness (log10 尺度，稳健计算)
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
    """Bowley（分位数）偏度；IQR=0 时返回 0.0"""
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    return float((q3 + q1 - 2.0 * q2) / iqr)


def _safe_skew(x: np.ndarray, method: str = "auto") -> Tuple[float, str]:
    """
    计算偏度，返回 (skew_value, method_used)。
    - method="auto": 若样本近似常数或矩法不稳定 → 回退到 Bowley
    - method="moment": 一直用矩法
    - method="quantile": 一直用 Bowley
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

    # auto: 先尝试矩法，若数值不稳定则回退
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
    计算时间分布结构相关指标
    
    返回列：
    - 分组列
    - Count: 样本数
    - Range: log10 尺度的范围
    - StdDev: log10 尺度的标准差
    - Skewness: log10 尺度的偏度
    - Skew_method: 偏度计算方法
    - Estimated_Peaks: GMM 估计的峰数
    - Ratio(Max/Pq_nonzero): 最大值与正值分位数的比值
    """
    # 分组器
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

        # 基础统计
        v_max = float(vals.max())
        v_min = float(vals.min())

        # 正值样本分位数（用于计算比率）
        positives = vals[vals > 0]
        if len(positives) > 0:
            p_nonzero = float(np.quantile(positives, ratio_quantile))
            if p_nonzero <= 0:
                p_nonzero = np.nan
        else:
            p_nonzero = np.nan
        ratio_val = (v_max / p_nonzero) if (p_nonzero is not np.nan) and np.isfinite(p_nonzero) else np.nan

        # log10 尺度的统计量
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

        # 若 log_std 未定义（正值太少），则退回到原始值
        if math.isnan(log_std):
            log_std = float(vals.std()) if len(vals) >= 2 else float("nan")
        if math.isnan(log_rng):
            log_rng = float(v_max - v_min)

        # GMM 峰数估计
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

        # 构建结果行
        row: Dict[str, object] = {
            "Count": int(n),
            "Range": log_rng,
            "StdDev": log_std,
            "Skewness": float(skew_val),
            "Skew_method": skew_used,
            "Estimated_Peaks": est_peaks,
            f"Ratio(Max/P{int(ratio_quantile*100):02d}_nonzero)": ratio_val,
        }

        # 添加分组列
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

    # 列顺序：分组列在前
    front_cols = (group_cols or ["Group"])
    metric_cols = [c for c in out.columns if c not in front_cols]
    return out[front_cols + metric_cols]


def _parse_group_col(arg: str) -> Optional[Union[str, List[str]]]:
    """解析命令行的 group_col 参数"""
    if arg is None:
        return "Continent"
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
        description="计算时间分布结构相关指标（Range, StdDev, Skewness, Estimated_Peaks）"
    )
    parser.add_argument("--csv_path", required=True, type=Path, help="输入 CSV 路径")
    parser.add_argument("--out_csv", type=Path, default=None, help="输出 CSV 路径")
    parser.add_argument("--group_col", type=str, default="Continent",
                        help='分组列：单列如 "Continent"，或多列如 "Continent,Country"；传 "none" 表示不分组')
    parser.add_argument("--tmrca_col", type=str, default="Time_years", help="TMRCA 数值列名")
    parser.add_argument("--ratio_quantile", type=float, default=0.01, help="用于分母的正值样本分位数（0-1）")
    parser.add_argument("--gmm_max_components", type=int, default=5, help="GMM 最大混合数（用于峰数估计）")
    parser.add_argument("--gmm_min_samples", type=int, default=10, help="启用 GMM 的最小样本数门槛")
    parser.add_argument("--random_state", type=int, default=42, help="GMM 随机种子")
    parser.add_argument("--skew_method", type=str, default="auto", choices=["auto", "moment", "quantile"],
                        help="偏度计算方法")

    args = parser.parse_args()

    # 校验
    if not args.csv_path.exists():
        raise FileNotFoundError(f"输入 CSV 未找到：{args.csv_path}")
    if not (0.0 < args.ratio_quantile < 1.0):
        raise ValueError(f"--ratio_quantile 必须在 (0,1) 内：当前 {args.ratio_quantile}")
    if args.gmm_max_components < 1:
        raise ValueError("--gmm_max_components 必须 >= 1")
    if args.gmm_min_samples < 0:
        raise ValueError("--gmm_min_samples 必须 >= 0")

    group_col = _parse_group_col(args.group_col)
    df = pd.read_csv(args.csv_path)

    # 计算
    res = calc_time_distribution_stats(
        df,
        group_col=group_col,
        tmrca_col=args.tmrca_col,
        ratio_quantile=float(args.ratio_quantile),
        gmm_max_components=int(args.gmm_max_components),
        gmm_min_samples=int(args.gmm_min_samples),
        random_state=int(args.random_state),
        skew_method=args.skew_method,
    )

    # 打印
    if res.empty:
        print("结果为空（可能样本过少或列名不匹配）。")
    else:
        if have_tabulate:
            print(tabulate(res, headers="keys", tablefmt="github", showindex=False))
        else:
            print(res.to_string(index=False))

    # 写出
    out_csv: Path
    if args.out_csv is not None:
        out_csv = args.out_csv
    else:
        out_csv = args.csv_path.parent / "time_distribution_stats.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n已保存结果到: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
