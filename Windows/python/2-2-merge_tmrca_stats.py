#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并时间深度和时间分布指标
功能：将 1-time_depth.py 和 1-time_distribution.py 的输出合并为一个完整的 CSV
"""

from pathlib import Path
from typing import Union, List, Optional

import argparse
import pandas as pd


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


def merge_tmrca_stats(
    time_depth_csv: Path,
    time_distribution_csv: Path,
    output_csv: Path,
    group_col: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    合并时间深度和时间分布统计结果
    
    参数：
    - time_depth_csv: 时间深度指标文件路径
    - time_distribution_csv: 时间分布指标文件路径
    - output_csv: 输出文件路径
    - group_col: 分组列名（用于合并）
    
    返回：
    - 合并后的 DataFrame
    """
    # 读取两个文件
    df_depth = pd.read_csv(time_depth_csv)
    df_dist = pd.read_csv(time_distribution_csv)
    
    # 确定分组列
    if group_col is None:
        merge_on = ["Group"]
    else:
        if isinstance(group_col, str):
            merge_on = [group_col]
        else:
            merge_on = list(group_col)
    
    # 检查分组列是否存在
    for col in merge_on:
        if col not in df_depth.columns:
            raise ValueError(f"时间深度文件缺少分组列：{col}")
        if col not in df_dist.columns:
            raise ValueError(f"时间分布文件缺少分组列：{col}")
    
    # 合并（外连接，保留所有分组）
    # Count 列在两个文件中都有，保留第一个即可
    df_dist_cols = [c for c in df_dist.columns if c not in merge_on and c != "Count"]
    df_merged = df_depth.merge(
        df_dist[merge_on + df_dist_cols],
        on=merge_on,
        how="outer"
    )
    
    # 保存
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    return df_merged


def main():
    try:
        from tabulate import tabulate
        have_tabulate = True
    except Exception:
        have_tabulate = False

    parser = argparse.ArgumentParser(
        description="合并时间深度和时间分布指标为完整的 TMRCA 统计结果"
    )
    parser.add_argument("--time_depth_csv", required=True, type=Path,
                        help="时间深度指标文件路径（1-time_depth.py 的输出）")
    parser.add_argument("--time_distribution_csv", required=True, type=Path,
                        help="时间分布指标文件路径（1-time_distribution.py 的输出）")
    parser.add_argument("--out_csv", required=True, type=Path,
                        help="输出合并后的 CSV 文件路径")
    parser.add_argument("--group_col", type=str, default="Continent",
                        help='分组列：单列如 "Continent"，或多列如 "Continent,Country"；传 "none" 表示不分组')

    args = parser.parse_args()

    # 校验
    if not args.time_depth_csv.exists():
        raise FileNotFoundError(f"时间深度文件未找到：{args.time_depth_csv}")
    if not args.time_distribution_csv.exists():
        raise FileNotFoundError(f"时间分布文件未找到：{args.time_distribution_csv}")

    group_col = _parse_group_col(args.group_col)

    # 合并
    result = merge_tmrca_stats(
        args.time_depth_csv,
        args.time_distribution_csv,
        args.out_csv,
        group_col=group_col,
    )

    # 打印
    if result.empty:
        print("合并结果为空。")
    else:
        if have_tabulate:
            print(tabulate(result, headers="keys", tablefmt="github", showindex=False))
        else:
            print(result.to_string(index=False))
        print(f"\n已保存合并结果到: {args.out_csv.resolve()}")
        print(f"总计 {len(result)} 个分组")


if __name__ == "__main__":
    main()
