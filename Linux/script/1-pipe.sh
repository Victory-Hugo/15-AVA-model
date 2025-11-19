#!/usr/bin/env bash
set -euo pipefail

# ==== Path ====
ROOT="/mnt/f/OneDrive/文档（科研）/脚本/Download/15-AVA-model/Linux/"
PY="$ROOT/python"
DATA="$ROOT/Example/input"
OUT="$ROOT/output"

# ==== Keys ====
UNIQUE_KEYS="Africa,Central_Asia,Southeast_Asia"

mkdir -p "$OUT"

# 1) 
python3 "$PY/1-divergence_time_shape.py" \
  --csv_path "$DATA/1-ID_Time_Class.csv" \
  --out_csv   "$OUT/Final_tmrca_stats.csv" \
  --group_col Continent \
  --tmrca_col Time_years \
  --ancient_threshold 100000 \
  --ratio_quantile 0.01 \
  --gmm_max_components 5 \
  --gmm_min_samples 10 \
  --random_state 42 \
  --skew_method auto

# 2) 
python3 "$PY/2-AMOVA.py" \
  --class_col "Continent" \
  --variation_type "Source of variation" \
  --variation_value "Percentage of variation" \
  --input  "$DATA/2-AMOVA.csv" \
  --output "$OUT/Final_AMOVA_scores.csv"

# 3) 
python3 "$PY/3-unique_haplogroup.py" \
  --csv "$DATA/3-public.csv" \
  --class-col "Continent" \
  --keyword "$UNIQUE_KEYS" \
  --out-dir "$OUT/"

# 4) 
python3 "$PY/4-score.py" \
  --tmrca  "$OUT/Final_tmrca_stats.csv" \
  --amova  "$OUT/Final_AMOVA_scores.csv" \
  --unique "$OUT/Final_unique_hap.csv" \
  --out    "$OUT/Final_metrics_scored.csv" \
  --w-max 2 --w-ancient 2 \
  --w-std 1 --w-range 1 --w-skew 1 --w-peaks 1 \
  --w-diversity 1.5 \
  --w-unique 1.5 \
  --thr-origin 0.7 \
  --thr-mix-low 0.3
