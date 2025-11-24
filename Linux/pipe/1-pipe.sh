#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Fixed root paths and basic configuration
###############################################################################

PY="15-AVA-model/Linux/python"

IN_TIME_CSV="15-AVA-model/Linux/input/Time.csv"
IN_AMOVA_CSV="15-AVA-model/Linux/input/AMOVA.csv"
IN_UNIQ_CSV="15-AVA-model/Linux/input/public.csv"
OUT="15-AVA-model/Linux/output"
# Fixed uniqueness keywords (same as original command)
KEY_COLMN="River"
UNIQUE_KEYS="Yellow River,Yangtze River,Zhujiang River"
TIME_COLMN="Time_years"

###############################################################################
# Time-threshold sweep toggle (hard-coded)
###############################################################################
# YES: enable time-threshold sweep
# NO: disable time-threshold sweep (default)
ENABLE_TIME_VARY_ANALYSIS="YES"

# Time-threshold sweep parameters (only effective when enabled)
ANCIENT_START=25000
ANCIENT_END=0
ANCIENT_STEP=500
KERNEL_SIGMA=0.4
TD_SIGMA=0.1


mkdir -p "$OUT"

###############################################################################
# Main workflow: fixed ancient_threshold=50000
###############################################################################
echo "[INFO] ===== Starting main workflow (ancient_threshold=50000) ====="

# 1a) Time depth metric calculation
echo "[INFO] 1a) Calculating time depth metrics..."
python3 "$PY/1-time_depth.py" \
  --csv_path "$IN_TIME_CSV" \
  --out_csv   "$OUT/time_depth_stats.csv" \
  --group_col "${KEY_COLMN}" \
  --tmrca_col "${TIME_COLMN}" \
  --ancient_threshold 50000 \
  --ratio_quantile 0.01 \
  --kernel_sigma 0.4 \
  --time_depth_sigma_log10 0.1

# 1b) Time distribution metric calculation
echo "[INFO] 1b) Calculating time distribution metrics..."
python3 "$PY/2-1-time_distribution.py" \
  --csv_path "$IN_TIME_CSV" \
  --out_csv   "$OUT/time_distribution_stats.csv" \
  --group_col "${KEY_COLMN}" \
  --tmrca_col "${TIME_COLMN}" \
  --ratio_quantile 0.01 \
  --gmm_max_components 5 \
  --gmm_min_samples 10 \
  --random_state 42 \
  --skew_method auto

# 1c) Merge time depth and time distribution metrics
echo "[INFO] 1c) Merging time depth and time distribution metrics..."
python3 "$PY/2-2-merge_tmrca_stats.py" \
  --time_depth_csv "$OUT/time_depth_stats.csv" \
  --time_distribution_csv "$OUT/time_distribution_stats.csv" \
  --out_csv "$OUT/Final_tmrca_stats.csv" \
  --group_col "${KEY_COLMN}"

# 2) AMOVA processing
echo "[INFO] 2) Computing AMOVA scores..."
python3 "$PY/3-AMOVA.py" \
  --class_col "${KEY_COLMN}" \
  --variation_type "Source of variation" \
  --variation_value "Percentage of variation" \
  --input  "$IN_AMOVA_CSV" \
  --output "$OUT/Final_AMOVA_scores.csv"

# 3) Haplogroup uniqueness
echo "[INFO] 3) Computing haplogroup uniqueness..."
python3 "$PY/4-Unique.py" \
  --csv "$IN_UNIQ_CSV" \
  --class-col "${KEY_COLMN}" \
  --keyword "$UNIQUE_KEYS" \
  --out-dir "$OUT/"

# 4) Scoring (four primary dimensions)
echo "[INFO] 4) Scoring (four primary dimensions)..."
python3 "$PY/5-score.py" \
  --tmrca  "$OUT/Final_tmrca_stats.csv" \
  --amova  "$OUT/Final_AMOVA_scores.csv" \
  --unique "$OUT/Final_unique_hap.csv" \
  --out    "$OUT/Final_metrics_scored.csv" \
  --group-col "${KEY_COLMN}" \
  --w-time-depth 4.0 \
  --w-time-structure 2.0 \
  --w-diversity 1 \
  --w-unique 1 \
  --thr-origin 0.7 \
  --thr-mix-low 0.3

echo "[INFO] ===== Main workflow complete ====="

###############################################################################
# Time-threshold sweep analysis (optional)
###############################################################################
if [[ "$ENABLE_TIME_VARY_ANALYSIS" == "YES" ]]; then
  echo ""
  echo "[INFO] ===== Starting time-threshold sweep analysis ====="
  
  TIME_DIR="$OUT/不同时间段"
  mkdir -p "$TIME_DIR"

  # Parameter validation
  if (( ANCIENT_STEP <= 0 )); then
    echo "[ERROR] ANCIENT_STEP must be a positive integer; current value is ${ANCIENT_STEP}" >&2
    exit 1
  fi
  if (( ANCIENT_START < ANCIENT_END )); then
    echo "[ERROR] Require ANCIENT_START >= ANCIENT_END" >&2
    exit 1
  fi

  COMMON_AMOVA="$TIME_DIR/Final_AMOVA_scores.csv"
  COMMON_UNIQUE="$TIME_DIR/Final_unique_hap.csv"
  AGG_CSV="$TIME_DIR/time_vary_metrics.csv"
  PLOT_PNG="$TIME_DIR/time_vary_metrics.png"

  # Precompute threshold-independent files: AMOVA and Unique Hap
  echo "[INFO] Copying AMOVA scores into time-threshold analysis directory..."
  cp "$OUT/Final_AMOVA_scores.csv" "$COMMON_AMOVA"

  echo "[INFO] Copying haplogroup uniqueness into time-threshold analysis directory..."
  cp "$OUT/Final_unique_hap.csv" "$COMMON_UNIQUE"

  # Main loop: iterate ancient_threshold
  echo "[INFO] ancient_threshold decreases from ${ANCIENT_START} to ${ANCIENT_END} with step ${ANCIENT_STEP}"
  for thr in $(seq "$ANCIENT_START" "-$ANCIENT_STEP" "$ANCIENT_END"); do
    thr_dir="$TIME_DIR/threshold_${thr}"
    mkdir -p "$thr_dir"

    time_depth_csv="$thr_dir/time_depth_stats_thr_${thr}.csv"
    time_dist_csv="$thr_dir/time_distribution_stats_thr_${thr}.csv"
    tmrca_csv="$thr_dir/Final_tmrca_stats_thr_${thr}.csv"
    score_csv="$thr_dir/Final_metrics_scored_thr_${thr}.csv"

    if [[ -f "$score_csv" ]]; then
      echo "[SKIP] Threshold ${thr} already has results; skip computation."
      continue
    fi

    echo "[INFO] >>> Threshold ${thr} years: time depth analysis"
    python3 "$PY/1-time_depth.py" \
      --csv_path "$DATA/Time.csv" \
      --out_csv   "$time_depth_csv" \
      --group_col "${KEY_COLMN}" \
      --tmrca_col "${TIME_COLMN}" \
      --ancient_threshold "$thr" \
      --ratio_quantile 0.01 \
      --kernel_sigma "$KERNEL_SIGMA" \
      --time_depth_sigma_log10 "$TD_SIGMA"

    echo "[INFO] >>> Threshold ${thr} years: time distribution analysis"
    python3 "$PY/2-1-time_distribution.py" \
      --csv_path "$DATA/Time.csv" \
      --out_csv   "$time_dist_csv" \
      --group_col "${KEY_COLMN}" \
      --tmrca_col "${TIME_COLMN}" \
      --ratio_quantile 0.01 \
      --gmm_max_components 5 \
      --gmm_min_samples 10 \
      --random_state 42 \
      --skew_method auto

    echo "[INFO] >>> Threshold ${thr} years: merge time metrics"
    python3 "$PY/2-2-merge_tmrca_stats.py" \
      --time_depth_csv "$time_depth_csv" \
      --time_distribution_csv "$time_dist_csv" \
      --out_csv "$tmrca_csv" \
      --group_col "${KEY_COLMN}"

    echo "[INFO] >>> Threshold ${thr} years: aggregate scores"
    python3 "$PY/5-score.py" \
      --tmrca  "$tmrca_csv" \
      --amova  "$COMMON_AMOVA" \
      --unique "$COMMON_UNIQUE" \
      --out    "$score_csv" \
      --group-col "${KEY_COLMN}" \
      --w-time-depth 4.0 \
      --w-time-structure 2.0 \
      --w-diversity 1 \
      --w-unique 1 \
      --thr-origin 0.7 \
      --thr-mix-low 0.3
  done

  # Aggregate all threshold results
  echo "[INFO] Aggregating all threshold results to ${AGG_CSV}"
  python3 "$PY/6-aggregate_time_vary_results.py" "$TIME_DIR" "$AGG_CSV" "${KEY_COLMN}"

  # Visualization: show multi-dimensional scores for regions across thresholds
  echo "[INFO] Drawing visualization to ${PLOT_PNG}"
  python3 "$PY/7-plot_time_vary_metrics.py" "$AGG_CSV" "$PLOT_PNG" "${KEY_COLMN}"

  echo "[INFO] ===== Time-threshold sweep analysis complete ====="
else
  echo ""
  echo "[INFO] Time-threshold sweep analysis is disabled."
  echo "[INFO] To enable, set ENABLE_TIME_VARY_ANALYSIS=\"YES\" in this script."
fi

echo ""
echo "[DONE] All workflows complete."
