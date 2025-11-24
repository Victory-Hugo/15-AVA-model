# Create a robust script that reads an AMOVA CSV/TSV and writes a two-column CSV: <class_col>,Diversity_pattern_score
from pathlib import Path

"""
AMOVA -> Diversity_pattern_score (Scheme 1: Linear baseline)

公式：
Diversity_pattern_score = (Within%)/100

- 输入：包含分组列（--class_col）、“Source of variation”列、以及“Percentage of variation”列的 CSV/TSV（分隔符自动识别）。
- 逻辑：按分类列（--class_col）选取 Source of variation == "Within populations" 的百分数作为 Within%，线性映射到 [0,1]。
- 输出：两列 CSV：<class_col>,Diversity_pattern_score。
- 仅实现线性基线；不做归一化、权重或其它派生指标。

用法示例：
python3 /mnt/f/6_起源地混合地/4-整合打分系统/python/3-AMOVA整理.py \
   --class_col "Continent" \
   --variation_type "Source of variation" \
   --variation_value "Percentage of variation" \
   --input /mnt/f/6_起源地混合地/4-整合打分系统/data/AMOVA.csv \
   --output /mnt/f/6_起源地混合地/4-整合打分系统/output/AMOVA_scores.csv
"""

import argparse
import sys
from typing import List, Optional
import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    """在 df 中宽松匹配用户提供的列名 target，返回实际列名或 None。"""
    if target in df.columns:
        return target
    lowmap = {c.lower().strip(): c for c in df.columns}
    t_low = target.lower().strip()
    if t_low in lowmap:
        return lowmap[t_low]

    # 更宽松：去掉空格与括号/百分号等符号后再匹配
    def _norm(x: str) -> str:
        x = x.lower().strip()
        for ch in ['(', ')', '%']:
            x = x.replace(ch, ' ')
        x = ' '.join(x.split())
        return x

    normmap = {_norm(c): c for c in df.columns}
    t_norm = _norm(target)
    if t_norm in normmap:
        return normmap[t_norm]
    return None


def _to_numeric_percent(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.strip()
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    return pd.to_numeric(s, errors='coerce')


def load_amova(path: str, class_col: str, var_type_col: str, var_value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python')

    region_col = _find_col(df, class_col)
    source_col = _find_col(df, var_type_col)
    pct_col    = _find_col(df, var_value_col)

    missing = [name for name, real in [
        (class_col, region_col),
        (var_type_col, source_col),
        (var_value_col, pct_col),
    ] if real is None]
    if missing:
        raise ValueError(f"无法识别以下列：{missing}；输入文件包含列：{list(df.columns)}")

    df['_Region'] = df[region_col].astype(str).str.strip()
    df['_Source'] = df[source_col].astype(str).str.strip().str.lower()
    df['_Pct'] = _to_numeric_percent(df[pct_col])
    df['_Region_colname'] = region_col  # 记录原始列名，便于输出
    return df


def compute_origin_scores(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    mask_within = df['_Source'].str.fullmatch(r'within populations', case=False, na=False)
    within = df.loc[mask_within, ['_Region', '_Pct']].copy()
    if within.empty:
        raise ValueError("未找到 'Within populations' 行，请检查输入数据及列名。")

    agg = within.groupby('_Region', as_index=False)['_Pct'].mean()
    origin = agg.copy()
    origin['Diversity_pattern_score'] = (origin['_Pct'] / 100.0).clip(lower=0.0, upper=1.0)

    out = origin.rename(columns={'_Region': class_col})[[class_col, 'Diversity_pattern_score']]
    return out.sort_values(class_col).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="AMOVA -> Diversity_pattern_score（线性基线）")
    ap.add_argument('--class_col', required=True, help='分类列名（例如：Continent/Region）')
    ap.add_argument('--variation_type', required=True, help='变异类型列名（例如：Source of variation）')
    ap.add_argument('--variation_value', required=True, help='变异百分比列名（例如：Percentage of variation）')
    ap.add_argument('--input', required=True, help='输入 AMOVA CSV/TSV 文件路径')
    ap.add_argument('--output', required=True, help='输出 CSV 文件路径')
    args = ap.parse_args()

    df = load_amova(args.input, args.class_col, args.variation_type, args.variation_value)
    out = compute_origin_scores(df, args.class_col)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
