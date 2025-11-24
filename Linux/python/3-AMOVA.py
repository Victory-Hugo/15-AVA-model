# Create a robust script that reads an AMOVA CSV/TSV and writes a two-column CSV: <class_col>,Diversity_pattern_score
from pathlib import Path

"""
AMOVA -> Diversity_pattern_score (Scheme 1: Linear baseline)

Formula:
Diversity_pattern_score = (Within%)/100

- Input: CSV/TSV (delimiter auto-detected) containing a grouping column (--class_col), a "Source of variation" column, and a "Percentage of variation" column.
- Logic: for the grouping column (--class_col), take the percentage where Source of variation == "Within populations" as Within%, map linearly to [0,1].
- Output: two-column CSV: <class_col>,Diversity_pattern_score.
- Only implements the linear baseline; no normalization, weighting, or other derived metrics.

Usage example:
python3 /mnt/f/6_origin_mix/4-integrated_scoring/python/3-AMOVA_clean.py \
   --class_col "Continent" \
   --variation_type "Source of variation" \
   --variation_value "Percentage of variation" \
   --input /mnt/f/6_origin_mix/4-integrated_scoring/data/AMOVA.csv \
   --output /mnt/f/6_origin_mix/4-integrated_scoring/output/AMOVA_scores.csv
"""

import argparse
import sys
from typing import List, Optional
import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    """Loosely match a user-supplied column name in df, return the actual name or None."""
    if target in df.columns:
        return target
    lowmap = {c.lower().strip(): c for c in df.columns}
    t_low = target.lower().strip()
    if t_low in lowmap:
        return lowmap[t_low]

    # Looser: strip spaces and symbols like parentheses/percent signs before matching
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
        raise ValueError(f"Could not identify these columns: {missing}; input file columns: {list(df.columns)}")

    df['_Region'] = df[region_col].astype(str).str.strip()
    df['_Source'] = df[source_col].astype(str).str.strip().str.lower()
    df['_Pct'] = _to_numeric_percent(df[pct_col])
    df['_Region_colname'] = region_col  # record the original column name for output
    return df


def compute_origin_scores(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    mask_within = df['_Source'].str.fullmatch(r'within populations', case=False, na=False)
    within = df.loc[mask_within, ['_Region', '_Pct']].copy()
    if within.empty:
        raise ValueError("No 'Within populations' row found; please check input data and column names.")

    agg = within.groupby('_Region', as_index=False)['_Pct'].mean()
    origin = agg.copy()
    origin['Diversity_pattern_score'] = (origin['_Pct'] / 100.0).clip(lower=0.0, upper=1.0)

    out = origin.rename(columns={'_Region': class_col})[[class_col, 'Diversity_pattern_score']]
    return out.sort_values(class_col).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="AMOVA -> Diversity_pattern_score (linear baseline)")
    ap.add_argument('--class_col', required=True, help='Grouping column name (e.g., Continent/Region)')
    ap.add_argument('--variation_type', required=True, help='Variation type column name (e.g., Source of variation)')
    ap.add_argument('--variation_value', required=True, help='Variation percentage column name (e.g., Percentage of variation)')
    ap.add_argument('--input', required=True, help='Input AMOVA CSV/TSV file path')
    ap.add_argument('--output', required=True, help='Output CSV file path')
    args = ap.parse_args()

    df = load_amova(args.input, args.class_col, args.variation_type, args.variation_value)
    out = compute_origin_scores(df, args.class_col)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
