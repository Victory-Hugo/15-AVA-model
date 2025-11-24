#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot line charts based on aggregated time-threshold sweep results:
- X-axis: ancient_threshold
- Y-axis: All_score
- One line per group
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
        raise ValueError(f"Aggregated file is missing grouping column: {group_col}")
    if "ancient_threshold" not in df.columns:
        raise ValueError("Aggregated file is missing ancient_threshold column")
    if "All_score" not in df.columns:
        raise ValueError("Aggregated file is missing All_score column")

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
    print(f"[OK] Plot complete: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot time-threshold sweep score line chart")
    parser.add_argument("agg_csv", type=Path, help="Output CSV from 6-aggregate_time_vary_results.py")
    parser.add_argument("out_png", type=Path, help="Output PNG path")
    parser.add_argument("group_col", type=str, help="Grouping column name (e.g., River / Continent)")
    args = parser.parse_args()

    plot_scores(args.agg_csv, args.out_png, args.group_col)


if __name__ == "__main__":
    main()
