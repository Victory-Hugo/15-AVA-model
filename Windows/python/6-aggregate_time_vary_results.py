#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate time-threshold sweep results:
- Scan threshold_* subdirectories under base_dir
- Read Final_metrics_scored_thr_*.csv files inside
- Append ancient_threshold column and concatenate vertically
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


def _parse_threshold(path: Path) -> Optional[float]:
    """Parse threshold number from file or parent directory name."""
    candidates: Sequence[str] = [
        path.stem,
        path.name,
        path.parent.name,
    ]
    for text in candidates:
        m = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def collect_metric_files(base_dir: Path) -> List[Path]:
    """Return list of all matched scoring result files."""
    pattern = "threshold_*/Final_metrics_scored_thr_*.csv"
    return sorted(base_dir.glob(pattern))


def aggregate(base_dir: Path, out_csv: Path, group_col: str) -> pd.DataFrame:
    """
    Aggregate multi-threshold score tables and output a long-format table with ancient_threshold column.
    """
    files = collect_metric_files(base_dir)
    if not files:
        raise FileNotFoundError(f"No threshold score files found under {base_dir}.")

    frames: List[pd.DataFrame] = []
    for file in files:
        thr = _parse_threshold(file)
        if thr is None:
            raise ValueError(f"Could not parse threshold from file name: {file}")
        df = pd.read_csv(file)
        if group_col not in df.columns:
            raise ValueError(f"{file} is missing grouping column: {group_col}")
        df = df.copy()
        df["ancient_threshold"] = thr
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(by=["ancient_threshold", group_col], ascending=[False, True])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate time-threshold sweep scoring results into a long-form CSV")
    parser.add_argument("base_dir", type=Path, help="Root directory containing threshold_* subdirectories")
    parser.add_argument("out_csv", type=Path, help="Output aggregated CSV path")
    parser.add_argument("group_col", type=str, help="Grouping column name (e.g., River / Continent)")
    args = parser.parse_args()

    combined = aggregate(args.base_dir, args.out_csv, args.group_col)
    print(f"[OK] Aggregated {len(combined)} rows to {args.out_csv}")
    print(f"Threshold range: {combined['ancient_threshold'].min():g} - {combined['ancient_threshold'].max():g}")


if __name__ == "__main__":
    main()
