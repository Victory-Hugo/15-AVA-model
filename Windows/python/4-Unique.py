#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：
- 支持一次性传入多个关键词（--keyword "A,B,C"），逐个执行相同逻辑：
  1) 导出 unique_haplogroups_<keyword>.txt
  2) 导出 merged_haplogroup_frequencies_<keyword>.txt（含总人数列）
  3) 导出打分结果文件 score_<keyword>.csv
  4) 汇总输出 Final_unique_hap.csv（两列：<class-col>, Unique_hap_score）

输出结构：
- <out-dir>/Final_unique_hap.csv
- <out-dir>/Frequency_result/
    ├─ unique_haplogroups_<keyword>.txt
    ├─ merged_haplogroup_frequencies_<keyword>.txt
    ├─ score_<keyword>.csv
    └─ combined_haplogroup_scores.csv

频率定义（样本频率，而非真实人口频率）：
# freq_in  = count_in  / total_in
# freq_out = count_out / total_out
"""

import argparse
import pandas as pd
from pathlib import Path
import re
from typing import List, Optional, Dict

# 中英文常见大洲映射（便捷用，可自行补充）
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
    """将关键字转为适合文件名的 token。"""
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.strip())

def split_keywords(kw_arg: str) -> List[str]:
    """从 --keyword 参数解析出多个关键词；以逗号分隔，剔除空白与空项。"""
    if not kw_arg:
        return []
    parts = [x.strip() for x in kw_arg.split(",")]
    return [x for x in parts if x]

def recalculate_unique_scores(data: pd.DataFrame, freq_col_in: str, freq_col_out: str) -> pd.DataFrame:
    """重新计算得分：仅依据“该类内>0且类外=0”为1，否则为0。"""
    data = data.copy()
    data['score'] = ((data[freq_col_in] > 0) & (data[freq_col_out] == 0)).astype(int)
    return data

def generate_combined_score_file(all_scores: List[dict], all_haplogroups: List[str], freq_dir: Path) -> None:
    """生成综合得分文件，包含每个大陆的每种单倍群得分和每个大陆的平均分（写入 Frequency_result/）"""
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

    # 每个大陆的“相对于全体单倍群”的平均分（保留原有逻辑）
    avg_row = {'Haplogroup': 'AVERAGE'}
    for continent in continent_names:
        avg_score = (continent_totals[continent] / continent_counts[continent]) if continent_counts[continent] > 0 else 0.0
        avg_row[f'Score_{continent}'] = avg_score

    combined_df = pd.concat([combined_df, pd.DataFrame([avg_row])], ignore_index=True)

    freq_dir.mkdir(parents=True, exist_ok=True)
    combined_path = freq_dir / "combined_haplogroup_scores.csv"
    combined_df.to_csv(combined_path, index=False, float_format="%.6f")
    print(f"[OK] 综合得分文件: {combined_path}")
    print(f"      包含 {len(all_haplogroups)} 个单倍群，{len(continent_names)} 个分类: {', '.join(continent_names)}")
    for continent in continent_names:
        avg_score = (continent_totals[continent] / continent_counts[continent]) if continent_counts[continent] > 0 else 0.0
        print(f"      {continent} 平均分(基于全体单倍群): {avg_score:.6f}")

def compute_and_export_for_keyword(
    df: pd.DataFrame,
    kw_raw: str,
    class_col: str,
    haplogroup_col: str,
    freq_dir: Path,
    case_insensitive: bool = True,
) -> Optional[dict]:
    """对单个 keyword 执行：筛选、频率计算、导出两个文件以及打分文件，返回得分数据（文件写入 Frequency_result/）"""
    # 关键字映射（支持中文 -> 英文），映射不到则原样使用
    kw = ZH2EN.get(kw_raw, kw_raw)
    token = sanitize_token(kw)

    if class_col not in df.columns:
        raise ValueError(f"分类列 {class_col} 不在数据中。现有列：{list(df.columns)}")
    if haplogroup_col not in df.columns:
        raise ValueError(f"单倍群列 {haplogroup_col} 不在数据中。现有列：{list(df.columns)}")

    class_series = df[class_col].astype(str)
    hap_series = df[haplogroup_col].astype(str)

    # 包含匹配（默认大小写不敏感）
    mask_in = class_series.str.contains(kw, case=not case_insensitive, na=False)
    df_in = df[mask_in].copy()
    df_out = df[~mask_in].copy()

    total_in = len(df_in)
    total_out = len(df_out)

    if total_in == 0:
        print(f"[WARN] 关键词“{kw_raw}”({kw})：筛选结果为空，跳过。")
        return None

    # 该分类下的单倍群（按字母排序）
    unique_haps = sorted(df_in[haplogroup_col].dropna().astype(str).unique())

    # 频率计算
    counts_in = df_in[haplogroup_col].value_counts()
    freq_in = (counts_in / float(total_in)).reindex(unique_haps, fill_value=0.0)

    if total_out == 0:
        freq_out = pd.Series(0.0, index=unique_haps)
    else:
        counts_out = df_out[haplogroup_col].value_counts()
        freq_out = (counts_out / float(total_out)).reindex(unique_haps, fill_value=0.0)

    # 合并频率表 + 总人数列
    merged = pd.DataFrame({
        "Haplogroup": unique_haps,
        f"Frequency_{token}": freq_in.values,
        f"Frequency_non_{token}": freq_out.values,
        f"Total_{token}": total_in,
        f"Total_non_{token}": total_out
    })

    # --- 导出（至 Frequency_result/） ---
    freq_dir.mkdir(parents=True, exist_ok=True)

    # 导出独特单倍群列表
    unique_path = freq_dir / f"unique_haplogroups_{token}.txt"
    with unique_path.open("w", encoding="utf-8") as f:
        for h in unique_haps:
            f.write(f"{h}\n")

    # 导出合并频率表
    merged_path = freq_dir / f"merged_haplogroup_frequencies_{token}.txt"
    merged.to_csv(merged_path, sep="\t", index=False, float_format="%.6f")

    # 计算独特单倍群得分（在 merged 上操作）
    merged_scored = recalculate_unique_scores(merged, f"Frequency_{token}", f"Frequency_non_{token}")

    # 导出打分文件（逐 haplogroup）
    score_path = freq_dir / f"score_{token}.csv"
    merged_scored[['Haplogroup', 'score']].to_csv(score_path, index=False)

    print(f"[OK] {kw_raw} -> 独特单倍群: {unique_path}")
    print(f"[OK] {kw_raw} -> 合并频率表: {merged_path}")
    print(f"[OK] {kw_raw} -> 打分结果: {score_path}")
    print(f"      样本数：in={total_in}, out={total_out}；单倍群数（in）={len(unique_haps)}")

    # 该分类的 Unique_hap_score（只基于该分类“出现过”的单倍群）
    if len(merged_scored) > 0:
        unique_ratio = float(merged_scored['score'].mean())  # ∈ [0,1]
    else:
        unique_ratio = 0.0

    # 返回供后续合并使用
    return {
        'continent': sanitize_token(kw),                # 用于 combined 文件列名拼接
        'label_for_final': kw,                          # 用于 Final_unique_hap.csv 的显示
        'scores': merged_scored[['Haplogroup', 'score']],
        'unique_ratio': unique_ratio
    }

def main():
    parser = argparse.ArgumentParser(description="按分类标准（支持多关键字）导出单倍群列表与合并频率表。")
    parser.add_argument("--csv", required=True, help="输入 CSV 文件路径（需含列：Continent, Haplogroup）")
    parser.add_argument("--class-col", default="Continent", help="分类列名（默认 Continent）")
    parser.add_argument("--keyword", required=True,
                        help='分类关键字，支持多个，用逗号分隔，例如 "Central_Asia,Southeast_Asia"；也可用中文，如 "中亚,东南亚"')
    parser.add_argument("--hap-col", default="Haplogroup", help="单倍群列名（默认 Haplogroup）")
    parser.add_argument("--out-dir", default="./outputs", help="输出目录（默认 ./outputs）")
    parser.add_argument("--case-sensitive", action="store_true", help="大小写敏感匹配（默认不敏感）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    freq_dir = out_dir / "Frequency_result"  # 新增：其它输出写到该子目录
    df = pd.read_csv(args.csv)

    keywords = split_keywords(args.keyword)
    if not keywords:
        raise ValueError("未检测到有效关键字，请用逗号分隔传入，例如：--keyword \"Central_Asia,Southeast_Asia\"")

    # 去重但保持顺序
    seen = set()
    deduped_keywords = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            deduped_keywords.append(k)

    # 收集所有得分数据用于合并
    all_scores = []
    all_unique_haplogroups = set()

    # 收集用于 Final_unique_hap.csv 的行
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
                args.class_col: score_data['label_for_final'],  # 列名等于 --class-col
                'Unique_hap_score': max(0.0, min(1.0, score_data['unique_ratio']))
            })

    # 生成综合得分文件（写入 Frequency_result/）
    if all_scores:
        generate_combined_score_file(all_scores, list(all_unique_haplogroups), freq_dir)

    # 输出 Final_unique_hap.csv（直接写在 out_dir/）
    if final_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        final_df = pd.DataFrame(final_rows, columns=[args.class_col, 'Unique_hap_score'])
        final_path = out_dir / "Final_unique_hap.csv"
        final_df.to_csv(final_path, index=False, float_format="%.6f")
        print(f"[OK] 已输出 Final_unique_hap.csv: {final_path}")
    else:
        print("[WARN] 未生成 Final_unique_hap.csv（所有关键词均为空）。")

if __name__ == "__main__":
    main()
