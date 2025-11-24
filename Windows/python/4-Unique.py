#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Purpose:
- Support multiple keywords in one pass (--keyword "A,B,C"), apply the same logic to each:
  1) Export unique_haplogroups_<keyword>.txt
  2) Export merged_haplogroup_frequencies_<keyword>.txt (includes total counts)
  3) Export scoring result file score_<keyword>.csv
  4) Aggregate output Final_unique_hap.csv (two columns: <class-col>, Unique_hap_score)

Output structure:
- <out-dir>/Final_unique_hap.csv
- <out-dir>/Frequency_result/
    ├─ unique_haplogroups_<keyword>.txt
    ├─ merged_haplogroup_frequencies_<keyword>.txt
    ├─ score_<keyword>.csv
    └─ combined_haplogroup_scores.csv

Frequency definition (sample frequency, not real population frequency):
# freq_in  = count_in  / total_in
# freq_out = count_out / total_out
"""

import argparse
import pandas as pd
from pathlib import Path
import re
from typing import List, Optional, Dict

# Common Chinese-English continent mapping (convenience; extend as needed)
ZH2EN = {
    "非洲": "Africa",
    "欧洲": "Europe",
    "亚洲": "Asia",
    "大洋洲": "Oceania",
    "北美洲": "North_America",
    "南美洲": "South_America",
    "中亚": "Central_Asia",
    "东南亚": "Southeast_Asia",
    "西亚": "West_Asia",
    "南亚": "South_Asia",
    "东亚": "East_Asia",
    "中东": "Middle_East",
}

def sanitize_token(s: str) -> str:
    """Convert a keyword into a token suitable for file names."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.strip())

def split_keywords(kw_arg: str) -> List[str]:
    """Parse multiple keywords from the --keyword argument; comma-separated, trimming blanks and empty items."""
    if not kw_arg:
        return []
    parts = [x.strip() for x in kw_arg.split(",")]
    return [x for x in parts if x]

def recalculate_unique_scores(data: pd.DataFrame, freq_col_in: str, freq_col_out: str) -> pd.DataFrame:
    """Recalculate scores: only mark 1 when in-class > 0 and out-of-class = 0, otherwise 0."""
    data = data.copy()
    data['score'] = ((data[freq_col_in] > 0) & (data[freq_col_out] == 0)).astype(int)
    return data

def generate_combined_score_file(all_scores: List[dict], all_haplogroups: List[str], freq_dir: Path) -> None:
    """Generate combined score file with each continent's haplogroup scores and average (written to Frequency_result/)"""
    combined_data = []
    continent_names = [score_data['continent'] for score_data in all_scores]

    continent_totals: Dict[str, float] = {continent: 0.0 for continent in continent_names}
    continent_counts: Dict[str, int] = {continent: 0 for continent in continent_names}

    for hap in sorted(all_haplogroups):
        row = {'Haplogroup': hap}
        for score_data in all_scores:
            continent = score_data['continent']
            hap_scores = score_data['scores']
            if hap in hap_scores['Haplogroup'].values:
                score = float(hap_scores.loc[hap_scores['Haplogroup'] == hap, 'score'].iloc[0])
            else:
                score = 0.0
            row[f'Score_{continent}'] = score
            continent_totals[continent] += score
            continent_counts[continent] += 1
        combined_data.append(row)

    combined_df = pd.DataFrame(combined_data)

    # Average score for each continent relative to all haplogroups (keep original logic)
    avg_row = {'Haplogroup': 'AVERAGE'}
    for continent in continent_names:
        avg_score = (continent_totals[continent] / continent_counts[continent]) if continent_counts[continent] > 0 else 0.0
        avg_row[f'Score_{continent}'] = avg_score

    combined_df = pd.concat([combined_df, pd.DataFrame([avg_row])], ignore_index=True)

    freq_dir.mkdir(parents=True, exist_ok=True)
    combined_path = freq_dir / "combined_haplogroup_scores.csv"
    combined_df.to_csv(combined_path, index=False, float_format="%.6f")
    print(f"[OK] Combined score file: {combined_path}")
    print(f"      Includes {len(all_haplogroups)} haplogroups, {len(continent_names)} categories: {', '.join(continent_names)}")
    for continent in continent_names:
        avg_score = (continent_totals[continent] / continent_counts[continent]) if continent_counts[continent] > 0 else 0.0
        print(f"      {continent} average (across all haplogroups): {avg_score:.6f}")

def compute_and_export_for_keyword(
    df: pd.DataFrame,
    kw_raw: str,
    class_col: str,
    haplogroup_col: str,
    freq_dir: Path,
    case_insensitive: bool = True,
) -> Optional[dict]:
    """Execute for a single keyword: filter, compute frequencies, export two files plus scoring file, return score data (files written to Frequency_result/)"""
    # Keyword mapping (supports Chinese -> English); use original if not mapped
    kw = ZH2EN.get(kw_raw, kw_raw)
    token = sanitize_token(kw)

    if class_col not in df.columns:
        raise ValueError(f"Class column {class_col} not in data. Existing columns: {list(df.columns)}")
    if haplogroup_col not in df.columns:
        raise ValueError(f"Haplogroup column {haplogroup_col} not in data. Existing columns: {list(df.columns)}")

    class_series = df[class_col].astype(str)
    hap_series = df[haplogroup_col].astype(str)

    # Inclusive match (case-insensitive by default)
    mask_in = class_series.str.contains(kw, case=not case_insensitive, na=False)
    df_in = df[mask_in].copy()
    df_out = df[~mask_in].copy()

    total_in = len(df_in)
    total_out = len(df_out)

    if total_in == 0:
        print(f"[WARN] Keyword \"{kw_raw}\" ({kw}): filtered result is empty, skip.")
        return None

    # Haplogroups under this category (alphabetical)
    unique_haps = sorted(df_in[haplogroup_col].dropna().astype(str).unique())

    # Frequency calculation
    counts_in = df_in[haplogroup_col].value_counts()
    freq_in = (counts_in / float(total_in)).reindex(unique_haps, fill_value=0.0)

    if total_out == 0:
        freq_out = pd.Series(0.0, index=unique_haps)
    else:
        counts_out = df_out[haplogroup_col].value_counts()
        freq_out = (counts_out / float(total_out)).reindex(unique_haps, fill_value=0.0)

    # Merge frequency table + total counts
    merged = pd.DataFrame({
        "Haplogroup": unique_haps,
        f"Frequency_{token}": freq_in.values,
        f"Frequency_non_{token}": freq_out.values,
        f"Total_{token}": total_in,
        f"Total_non_{token}": total_out
    })

    # --- Export (to Frequency_result/) ---
    freq_dir.mkdir(parents=True, exist_ok=True)

    # Export unique haplogroup list
    unique_path = freq_dir / f"unique_haplogroups_{token}.txt"
    with unique_path.open("w", encoding="utf-8") as f:
        for h in unique_haps:
            f.write(f"{h}\n")

    # Export merged frequency table
    merged_path = freq_dir / f"merged_haplogroup_frequencies_{token}.txt"
    merged.to_csv(merged_path, sep="\t", index=False, float_format="%.6f")

    # Compute unique haplogroup scores (operate on merged)
    merged_scored = recalculate_unique_scores(merged, f"Frequency_{token}", f"Frequency_non_{token}")

    # Export scoring file (per haplogroup)
    score_path = freq_dir / f"score_{token}.csv"
    merged_scored[['Haplogroup', 'score']].to_csv(score_path, index=False)

    print(f"[OK] {kw_raw} -> unique haplogroups: {unique_path}")
    print(f"[OK] {kw_raw} -> merged frequency table: {merged_path}")
    print(f"[OK] {kw_raw} -> scoring result: {score_path}")
    print(f"      Samples: in={total_in}, out={total_out}; haplogroups (in)={len(unique_haps)}")

    # Unique_hap_score for this category (only based on haplogroups appearing in this category)
    if len(merged_scored) > 0:
        unique_ratio = float(merged_scored['score'].mean())  # ∈ [0,1]
    else:
        unique_ratio = 0.0

    # Return for subsequent merging
    return {
        'continent': sanitize_token(kw),                # Used when concatenating column names in combined file
        'label_for_final': kw,                          # Used for display in Final_unique_hap.csv
        'scores': merged_scored[['Haplogroup', 'score']],
        'unique_ratio': unique_ratio
    }

def main():
    parser = argparse.ArgumentParser(description="Export haplogroup list and merged frequency table by category (supports multiple keywords).")
    parser.add_argument("--csv", required=True, help="Input CSV path (must include columns: Continent, Haplogroup)")
    parser.add_argument("--class-col", default="Continent", help="Category column name (default Continent)")
    parser.add_argument("--keyword", required=True,
                        help='Category keywords; supports multiple, comma-separated, e.g., "Central_Asia,Southeast_Asia"; Chinese keywords are also supported')
    parser.add_argument("--hap-col", default="Haplogroup", help="Haplogroup column name (default Haplogroup)")
    parser.add_argument("--out-dir", default="./outputs", help="Output directory (default ./outputs)")
    parser.add_argument("--case-sensitive", action="store_true", help="Case-sensitive matching (default insensitive)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    freq_dir = out_dir / "Frequency_result"  # New: other outputs go to this subdirectory
    df = pd.read_csv(args.csv)

    keywords = split_keywords(args.keyword)
    if not keywords:
        raise ValueError("No valid keyword detected; pass comma-separated values, e.g., --keyword \"Central_Asia,Southeast_Asia\"")

    # Deduplicate while preserving order
    seen = set()
    deduped_keywords = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            deduped_keywords.append(k)

    # Collect all score data for merging
    all_scores = []
    all_unique_haplogroups = set()

    # Collect rows for Final_unique_hap.csv
    final_rows: List[Dict[str, object]] = []

    for kw in deduped_keywords:
        score_data = compute_and_export_for_keyword(
            df=df,
            kw_raw=kw,
            class_col=args.class_col,
            haplogroup_col=args.hap_col,
            freq_dir=freq_dir,
            case_insensitive=not args.case_sensitive,
        )
        if score_data is not None:
            all_scores.append(score_data)
            all_unique_haplogroups.update(score_data['scores']['Haplogroup'])
            final_rows.append({
                args.class_col: score_data['label_for_final'],  # Column name equals --class-col
                'Unique_hap_score': max(0.0, min(1.0, score_data['unique_ratio']))
            })

    # Generate combined score file (written to Frequency_result/)
    if all_scores:
        generate_combined_score_file(all_scores, list(all_unique_haplogroups), freq_dir)

    # Output Final_unique_hap.csv (directly under out_dir/)
    if final_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        final_df = pd.DataFrame(final_rows, columns=[args.class_col, 'Unique_hap_score'])
        final_path = out_dir / "Final_unique_hap.csv"
        final_df.to_csv(final_path, index=False, float_format="%.6f")
        print(f"[OK] Final_unique_hap.csv written: {final_path}")
    else:
        print("[WARN] Final_unique_hap.csv not generated (all keywords empty).")

if __name__ == "__main__":
    main()
