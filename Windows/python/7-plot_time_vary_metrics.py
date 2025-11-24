#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于聚合后的时间阈值遍历结果绘制折线图：
- 横轴 ancient_threshold
- 纵轴 All_score
- 每个分组一条线
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_scores(agg_csv: Path, out_png: Path, group_col: str) -> None:
    df = pd.read_csv(agg_csv)
    if group_col not in df.columns:
        raise ValueError(f"聚合文件缺少分组列：{group_col}")
    if "ancient_threshold" not in df.columns:
        raise ValueError("聚合文件缺少 ancient_threshold 列")
    if "All_score" not in df.columns:
        raise ValueError("聚合文件缺少 All_score 列")

    plt.figure(figsize=(10, 6), dpi=140)
    colors: Dict[str, str] = {}
    for idx, (grp, gdf) in enumerate(df.groupby(group_col)):
        gdf = gdf.sort_values("ancient_threshold")
        color = plt.cm.tab20(idx % 20)
        colors[grp] = color
        plt.plot(
            gdf["ancient_threshold"],
            gdf["All_score"],
            marker="o",
            label=str(grp),
            color=color,
        )

    plt.xlabel("ancient_threshold")
    plt.ylabel("All_score")
    plt.title("All_score vs ancient_threshold")
    plt.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[OK] 绘图完成: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制时间阈值遍历得分折线图")
    parser.add_argument("agg_csv", type=Path, help="6-aggregate_time_vary_results.py 的输出 CSV")
    parser.add_argument("out_png", type=Path, help="输出 PNG 路径")
    parser.add_argument("group_col", type=str, help="分组列名（如 River / Continent）")
    args = parser.parse_args()

    plot_scores(args.agg_csv, args.out_png, args.group_col)


if __name__ == "__main__":
    main()
