#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge time depth and time distribution metrics
Purpose: merge outputs of 1-time_depth.py and 1-time_distribution.py into a complete CSV
"""

from pathlib import Path
from typing import Union, List, Optional

import argparse
import pandas as pd


def _parse_group_col(arg: str) -> Optional[Union[str, List[str]]]:
    """Parse the group_col argument from the command line"""
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
    Merge time depth and time distribution statistics
    
    Parameters:
    - time_depth_csv: path to the time depth metrics file
    - time_distribution_csv: path to the time distribution metrics file
    - output_csv: output file path
    - group_col: grouping column name(s) used for merging
    
    Returns:
    - merged DataFrame
    """
    # Read both files
    df_depth = pd.read_csv(time_depth_csv)
    df_dist = pd.read_csv(time_distribution_csv)
    
    # Determine grouping columns
    if group_col is None:
        merge_on = ["Group"]
    else:
        if isinstance(group_col, str):
            merge_on = [group_col]
        else:
            merge_on = list(group_col)
    
    # Check whether grouping columns exist
    for col in merge_on:
        if col not in df_depth.columns:
            raise ValueError(f"Time depth file is missing grouping column: {col}")
        if col not in df_dist.columns:
            raise ValueError(f"Time distribution file is missing grouping column: {col}")
    
    # Merge (outer join, keep all groups)
    # Count column exists in both files; keep the first one
    df_dist_cols = [c for c in df_dist.columns if c not in merge_on and c != "Count"]
    df_merged = df_depth.merge(
        df_dist[merge_on + df_dist_cols],
        on=merge_on,
        how="outer"
    )
    
    # Save
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
        description="Merge time depth and time distribution metrics into complete TMRCA statistics"
    )
    parser.add_argument("--time_depth_csv", required=True, type=Path,
                        help="Path to time depth metrics file (output of 1-time_depth.py)")
    parser.add_argument("--time_distribution_csv", required=True, type=Path,
                        help="Path to time distribution metrics file (output of 1-time_distribution.py)")
    parser.add_argument("--out_csv", required=True, type=Path,
                        help="Output path for merged CSV")
    parser.add_argument("--group_col", type=str, default="Continent",
                        help='Grouping columns: single column like "Continent", or multiple like "Continent,Country"; pass "none" for no grouping')

    args = parser.parse_args()

    # Validation
    if not args.time_depth_csv.exists():
        raise FileNotFoundError(f"Time depth file not found: {args.time_depth_csv}")
    if not args.time_distribution_csv.exists():
        raise FileNotFoundError(f"Time distribution file not found: {args.time_distribution_csv}")

    group_col = _parse_group_col(args.group_col)

    # Merge
    result = merge_tmrca_stats(
        args.time_depth_csv,
        args.time_distribution_csv,
        args.out_csv,
        group_col=group_col,
    )

    # Print
    if result.empty:
        print("Merged result is empty.")
    else:
        if have_tabulate:
            print(tabulate(result, headers="keys", tablefmt="github", showindex=False))
        else:
            print(result.to_string(index=False))
        print(f"\nSaved merged result to: {args.out_csv.resolve()}")
        print(f"Total {len(result)} group(s)")


if __name__ == "__main__":
    main()
