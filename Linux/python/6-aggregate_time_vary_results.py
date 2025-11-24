#!/usr/bin/env python3
"""
Aggregate scoring results for all time thresholds
"""
import sys
import pandas as pd
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print("Usage: python 6-aggregate_time_vary_results.py <time_dir> <output_csv> <key_column>")
        sys.exit(1)
    
    time_dir = Path(sys.argv[1])
    agg_csv = Path(sys.argv[2])
    key_column = sys.argv[3]
    
    rows = []
    for csv_path in sorted(time_dir.glob("threshold_*/Final_metrics_scored_thr_*.csv")):
        parent = csv_path.parent.name
        if "_" not in parent:
            continue
        try:
            thr = int(parent.split("_", 1)[1])
        except ValueError:
            continue
        df = pd.read_csv(csv_path)
        df["ancient_threshold"] = thr
        rows.append(df)
    
    if not rows:
        raise SystemExit("[ERROR] No Final_metrics_scored_thr_*.csv files found; cannot aggregate.")
    
    combined = pd.concat(rows, ignore_index=True)
    combined.sort_values(["ancient_threshold", key_column], ascending=[False, True], inplace=True)
    combined.to_csv(agg_csv, index=False)
    print(f"[OK] Wrote {agg_csv} with {len(combined)} rows.")


if __name__ == "__main__":
    main()
