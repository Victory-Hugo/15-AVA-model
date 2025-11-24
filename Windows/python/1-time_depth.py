#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间深度指标计算模块
功能：计算与时间深度相关的指标
- Max, Min, Min_nonzero, Pq_nonzero
- AncientRatio
- TimeDepthKernel
- KernelScore (多个时间尺度)
"""

from pathlib import Path
from typing import Union, List, Optional, Dict

import argparse
import math
import numpy as np
import pandas as pd


def _format_scale_label(scale: float) -> str:
    """格式化时间尺度标签，便于用作列名"""
    text = f"{scale:g}"
    text = text.replace(".", "p").replace("-", "m")
    return text


def _compute_kernel_scores(
    values: np.ndarray,
    scales: List[float],
    sigma: float,
    total_count: int,
) -> Dict[str, float]:
    """计算多个时间尺度的核得分"""
    if sigma <= 0:
        raise ValueError("kernel_sigma 必须 > 0")
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
    """计算平滑时间深度指数（以 log10 时间为尺度）"""
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
    """解析逗号分隔的时间尺度列表"""
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
            raise ValueError(f"无法解析 kernel_scales 中的数值：{token!r}")
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
    计算时间深度相关指标
    
    返回列：
    - 分组列
    - Count: 样本数
    - Max: 最大 TMRCA
    - Min: 最小 TMRCA
    - Min_nonzero: 最小正值 TMRCA
    - Pq_nonzero: 正值样本的 ratio_quantile 分位数
    - AncientRatio(>threshold): 古老谱系比例
    - KernelScore_T*: 各时间尺度核得分
    - TimeDepthKernel_T*: 时间深度平滑指数
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

        # 正值样本分位数
        positives = vals[vals > 0]
        if len(positives) > 0:
            p_nonzero = float(np.quantile(positives, ratio_quantile))
            if p_nonzero <= 0:
                p_nonzero = np.nan
            min_nonzero = float(positives.min())
        else:
            p_nonzero = np.nan
            min_nonzero = np.nan

        # 古老谱系比例
        ancient_ratio = float((vals > ancient_threshold).mean())

        # 时间尺度核得分
        kernel_scores: Dict[str, float] = {}
        if kernel_scales:
            kernel_scores = _compute_kernel_scores(vals.values, kernel_scales, kernel_sigma, n)

        # 平滑时间深度指数
        td_focus = time_depth_focus or ancient_threshold
        if td_focus is not None and td_focus > 0:
            time_depth_kernel = _compute_time_depth_kernel(vals.values, float(td_focus), time_depth_sigma_log10)
        else:
            time_depth_kernel = 0.0

        # 构建结果行
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
        description="计算时间深度相关指标（Max, Min, AncientRatio, TimeDepthKernel, KernelScore）"
    )
    parser.add_argument("--csv_path", required=True, type=Path, help="输入 CSV 路径")
    parser.add_argument("--out_csv", type=Path, default=None, help="输出 CSV 路径")
    parser.add_argument("--group_col", type=str, default="Continent",
                        help='分组列：单列如 "Continent"，或多列如 "Continent,Country"；传 "none" 表示不分组')
    parser.add_argument("--tmrca_col", type=str, default="Time_years", help="TMRCA 数值列名")
    parser.add_argument("--ancient_threshold", type=float, default=100000.0, help="AncientRatio 阈值")
    parser.add_argument("--ratio_quantile", type=float, default=0.01, help="用于分母的正值样本分位数（0-1）")
    parser.add_argument("--kernel_scales", type=str, default="",
                        help="逗号分隔的时间尺度列表（默认自动取 ancient_threshold 的 0.1×、1×、10×）")
    parser.add_argument("--kernel_sigma", type=float, default=0.5,
                        help="log 尺度下高斯核的标准差 σ")
    parser.add_argument("--time_depth_focus", type=float, default=None,
                        help="时间深度平滑指标的焦点 T（默认等于 ancient_threshold）")
    parser.add_argument("--time_depth_sigma_log10", type=float, default=0.3,
                        help="时间深度高斯核的 log10 标准差")

    args = parser.parse_args()

    # 校验
    if not args.csv_path.exists():
        raise FileNotFoundError(f"输入 CSV 未找到：{args.csv_path}")
    if not (0.0 < args.ratio_quantile < 1.0):
        raise ValueError(f"--ratio_quantile 必须在 (0,1) 内：当前 {args.ratio_quantile}")

    group_col = _parse_group_col(args.group_col)
    df = pd.read_csv(args.csv_path)

    # 解析 kernel_scales
    if args.kernel_scales:
        kernel_scales = _parse_kernel_scales(args.kernel_scales)
    else:
        t = float(args.ancient_threshold)
        kernel_scales = [t / 10.0, t, t * 10.0]

    # 计算
    res = calc_time_depth_stats(
        df,
        group_col=group_col,
        tmrca_col=args.tmrca_col,
        ancient_threshold=float(args.ancient_threshold),
        ratio_quantile=float(args.ratio_quantile),
        kernel_scales=kernel_scales,
        kernel_sigma=float(args.kernel_sigma),
        time_depth_focus=float(args.time_depth_focus) if args.time_depth_focus else float(args.ancient_threshold),
        time_depth_sigma_log10=float(args.time_depth_sigma_log10),
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
        out_csv = args.csv_path.parent / "time_depth_stats.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n已保存结果到: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
