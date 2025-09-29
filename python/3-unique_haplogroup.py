#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function:
- Support passing multiple keywords at once (--keyword "A,B,C"), executing the same logic for each:
  1) Export unique_haplogroups_<keyword>.txt
  2)    format_status("EXPORT", f"{kw_raw} -> Unique haplogroups:", str(unique_path))
    format_status("EXPORT", f"{kw_raw} -> Merged frequency table:", str(merged_path))
    format_status("EXPORT", f"{kw_raw} -> Scoring results:", str(score_path))
    blue_print(f"         Sample counts: in={total_in}, out={total_out}; Haplogroup count (in)={len(unique_haps)}")ort merged_haplogroup_frequencies_<keyword>.txt (with total count column)
  3) Export scoring result file score_<keyword>.csv
  4) Summary output Final_unique_hap.csv (two columns: <class-col>, Unique_hap_score)

Frequency definition (sample frequency, not real population frequency):
# freq_in  = count_in  / total_in
# freq_out = count_out / total_out
"""

import argparse
import pandas as pd
from pathlib import Path
import re
from typing import List, Optional, Dict


# Blue color formatting for print statements
def blue_print(message, *args, **kwargs):
    """Print text in blue color with white variables for better readability"""
    # ANSI color codes
    BLUE = '\033[34m'
    WHITE = '\033[37m' 
    RESET = '\033[0m'
    
    if args or kwargs:
        # Format message with white variables
        formatted_msg = message.format(*args, **kwargs)
        # Replace variable placeholders with white color
        import re
        # Find {} placeholders and color them white
        result = re.sub(r'(\{[^}]*\})', f'{WHITE}\1{BLUE}', formatted_msg)
        print(f"{BLUE}{result}{RESET}")
    else:
        print(f"{BLUE}{message}{RESET}")

def format_status(status, message, value=""):
    """Format status messages with consistent structure"""
    BLUE = '\033[34m'
    WHITE = '\033[37m'
    GREEN = '\033[32m' 
    RESET = '\033[0m'
    
    if value:
        print(f"{GREEN}[{status}]{RESET} {BLUE}{message}{RESET} {WHITE}{value}{RESET}")
    else:
        print(f"{GREEN}[{status}]{RESET} {BLUE}{message}{RESET}")

# Common continent mapping between Chinese and English (convenient, can be extended)
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
    """Convert keyword to a file-name-safe token."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.strip())

def split_keywords(kw_arg: str) -> List[str]:
    """Parse multiple keywords from --keyword parameter; comma-separated, remove whitespace and empty items."""
    if not kw_arg:
        return []
    parts = [x.strip() for x in kw_arg.split(",")]
    return [x for x in parts if x]

def recalculate_unique_scores(data: pd.DataFrame, freq_col_in: str, freq_col_out: str) -> pd.DataFrame:
    """Recalculate scores: only "in-category >0 and out-category =0" gets 1, otherwise 0."""
    data = data.copy()
    data['score'] = ((data[freq_col_in] > 0) & (data[freq_col_out] == 0)).astype(int)
    return data

def generate_combined_score_file(all_scores: List[dict], all_haplogroups: List[str], freq_dir: Path) -> None:
    """Generate comprehensive score file, including each continent's score for each haplogroup and each continent's average score (written to Frequency_result/)"""
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

    # Each continent's "relative to all haplogroups" average score (keeping original logic)
    avg_row = {'Haplogroup': 'AVERAGE'}
    for continent in continent_names:
        avg_score = (continent_totals[continent] / continent_counts[continent]) if continent_counts[continent] > 0 else 0.0
        avg_row[f'Score_{continent}'] = avg_score

    combined_df = pd.concat([combined_df, pd.DataFrame([avg_row])], ignore_index=True)

    freq_dir.mkdir(parents=True, exist_ok=True)
    combined_path = freq_dir / "combined_haplogroup_scores.csv"
    combined_df.to_csv(combined_path, index=False, float_format="%.6f")
    format_status("SAVED", "Comprehensive score file:", str(combined_path))
    blue_print(f"         Contains {len(all_haplogroups)} haplogroups, {len(continent_names)} categories: {', '.join(continent_names)}")
    for continent in continent_names:
        avg_score = (continent_totals[continent] / continent_counts[continent]) if continent_counts[continent] > 0 else 0.0
        blue_print(f"         {continent} average score (based on all haplogroups): {avg_score:.6f}")

def compute_and_export_for_keyword(
    df: pd.DataFrame,
    kw_raw: str,
    class_col: str,
    haplogroup_col: str,
    freq_dir: Path,
    case_insensitive: bool = True,
) -> Optional[dict]:
    """For a single keyword: filter, calculate frequencies, export two files and scoring file, return scoring data (files written to Frequency_result/)"""
    # Keyword mapping (supports Chinese -> English), use original if no mapping found
    kw = ZH2EN.get(kw_raw, kw_raw)
    token = sanitize_token(kw)

    if class_col not in df.columns:
        raise ValueError(f"Classification column {class_col} not in data. Existing columns: {list(df.columns)}")
    if haplogroup_col not in df.columns:
        raise ValueError(f"Haplogroup column {haplogroup_col} not in data. Existing columns: {list(df.columns)}")

    class_series = df[class_col].astype(str)
    hap_series = df[haplogroup_col].astype(str)

    # Contains matching (case insensitive by default)
    mask_in = class_series.str.contains(kw, case=not case_insensitive, na=False)
    df_in = df[mask_in].copy()
    df_out = df[~mask_in].copy()

    total_in = len(df_in)
    total_out = len(df_out)

    if total_in == 0:
        blue_print(f"[WARN] Keyword '{kw_raw}' ({kw}): Filter result is empty, skipping.")
        return None

    # Haplogroups in this category (alphabetically sorted)
    unique_haps = sorted(df_in[haplogroup_col].dropna().astype(str).unique())

    # Frequency calculation
    counts_in = df_in[haplogroup_col].value_counts()
    freq_in = (counts_in / float(total_in)).reindex(unique_haps, fill_value=0.0)

    if total_out == 0:
        freq_out = pd.Series(0.0, index=unique_haps)
    else:
        counts_out = df_out[haplogroup_col].value_counts()
        freq_out = (counts_out / float(total_out)).reindex(unique_haps, fill_value=0.0)

    # Merge frequency table + total count column
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

    # Calculate unique haplogroup scores (operate on merged)
    merged_scored = recalculate_unique_scores(merged, f"Frequency_{token}", f"Frequency_non_{token}")

    # Export scoring file (per haplogroup)
    score_path = freq_dir / f"score_{token}.csv"
    merged_scored[['Haplogroup', 'score']].to_csv(score_path, index=False)

    blue_print(f"[OK] {kw_raw} -> Unique haplogroups: {unique_path}")
    blue_print(f"[OK] {kw_raw} -> Merged frequency table: {merged_path}")
    blue_print(f"[OK] {kw_raw} -> Scoring results: {score_path}")
    blue_print(f"     Sample counts: in={total_in}, out={total_out}; Haplogroup count (in)={len(unique_haps)}")

    # This category's Unique_hap_score (only based on haplogroups that "appeared" in this category)
    if len(merged_scored) > 0:
        unique_ratio = float(merged_scored['score'].mean())  # ∈ [0,1]
    else:
        unique_ratio = 0.0

    # Return for subsequent merging
    return {
        'continent': sanitize_token(kw),                # Used for combined file column name concatenation
        'label_for_final': kw,                          # Used for Final_unique_hap.csv display
        'scores': merged_scored[['Haplogroup', 'score']],
        'unique_ratio': unique_ratio
    }

def main():
    parser = argparse.ArgumentParser(description="Export haplogroup lists and merged frequency tables by classification criteria (supports multiple keywords).")
    parser.add_argument("--csv", required=True, help="Input CSV file path (must contain columns: Continent, Haplogroup)")
    parser.add_argument("--class-col", default="Continent", help="Classification column name (default: Continent)")
    parser.add_argument("--keyword", required=True,
                        help='Classification keywords, supports multiple, comma-separated, e.g. "Central_Asia,Southeast_Asia"; Chinese also supported, e.g. "中亚,东南亚"')
    parser.add_argument("--hap-col", default="Haplogroup", help="Haplogroup column name (default: Haplogroup)")
    parser.add_argument("--out-dir", default="./outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--case-sensitive", action="store_true", help="Case-sensitive matching (default: case-insensitive)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    freq_dir = out_dir / "Frequency_result"  # New: other outputs written to this subdirectory
    df = pd.read_csv(args.csv)

    keywords = split_keywords(args.keyword)
    if not keywords:
        raise ValueError("No valid keywords detected, please pass comma-separated, e.g.: --keyword \"Central_Asia,Southeast_Asia\"")

    # Remove duplicates but maintain order
    seen = set()
    deduped_keywords = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            deduped_keywords.append(k)

    # Collect all scoring data for merging
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

    # Generate comprehensive score file (written to Frequency_result/)
    if all_scores:
        generate_combined_score_file(all_scores, list(all_unique_haplogroups), freq_dir)

    # Output Final_unique_hap.csv (written directly to out_dir/)
    if final_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        final_df = pd.DataFrame(final_rows, columns=[args.class_col, 'Unique_hap_score'])
        final_path = out_dir / "Final_unique_hap.csv"
        final_df.to_csv(final_path, index=False, float_format="%.6f")
        format_status("SAVED", "Final unique haplotype scores:", str(final_path))
    else:
        blue_print("[WARN] Final_unique_hap.csv not generated (all keywords were empty).")

if __name__ == "__main__":
    main()
