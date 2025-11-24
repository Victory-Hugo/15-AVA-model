#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总时间阈值遍历结果：
- 扫描 base_dir 下的 threshold_* 子目录
- 读取其中的 Final_metrics_scored_thr_*.csv
- 追加 ancient_threshold 列并纵向合并
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


def _parse_threshold(path: Path) -> Optional[float]:
    """从文件或父目录名中解析阈值数字。"""
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
    """返回所有匹配的评分结果文件列表。"""
    pattern = "threshold_*/Final_metrics_scored_thr_*.csv"
    return sorted(base_dir.glob(pattern))


def aggregate(base_dir: Path, out_csv: Path, group_col: str) -> pd.DataFrame:
    """
    汇总多阈值得分表，输出含 ancient_threshold 列的长表格式。
    """
    files = collect_metric_files(base_dir)
    if not files:
        raise FileNotFoundError(f"未在 {base_dir} 下找到任何阈值评分文件。")

    frames: List[pd.DataFrame] = []
    for file in files:
        thr = _parse_threshold(file)
        if thr is None:
            raise ValueError(f"无法从文件名解析阈值：{file}")
        df = pd.read_csv(file)
        if group_col not in df.columns:
            raise ValueError(f"{file} 缺少分组列：{group_col}")
        df = df.copy()
        df["ancient_threshold"] = thr
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(by=["ancient_threshold", group_col], ascending=[False, True])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总时间阈值遍历的评分结果为长表 CSV")
    parser.add_argument("base_dir", type=Path, help="包含 threshold_* 子目录的根目录")
    parser.add_argument("out_csv", type=Path, help="输出聚合结果 CSV 路径")
    parser.add_argument("group_col", type=str, help="分组列名（如 River / Continent）")
    args = parser.parse_args()

    combined = aggregate(args.base_dir, args.out_csv, args.group_col)
    print(f"[OK] 汇总 {len(combined)} 行到 {args.out_csv}")
    print(f"阈值范围：{combined['ancient_threshold'].min():g} - {combined['ancient_threshold'].max():g}")


if __name__ == "__main__":
    main()
