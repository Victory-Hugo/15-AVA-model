#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 固定根路径与基本配置
###############################################################################

PY="F:/OneDrive/文档（科研）/脚本/Download/15-AVA-model/Linux/python"

IN_TIME_CSV="F:/OneDrive/文档（科研）/脚本/Download/15-AVA-model/Linux/input/1-mtDNA/Time.csv"
IN_AMOVA_CSV="F:/OneDrive/文档（科研）/脚本/Download/15-AVA-model/Linux/input/1-mtDNA/AMOVA.csv"
IN_UNIQ_CSV="F:/OneDrive/文档（科研）/脚本/Download/15-AVA-model/Linux/input/1-mtDNA/public.csv"
OUT="F:/OneDrive/文档（科研）/脚本/Download/15-AVA-model/Linux/output"
# 固定独特性关键词（与原始命令完全一致）
KEY_COLMN="Continent"
UNIQUE_KEYS="Africa,Central_Asia,Southeast_Asia"
TIME_COLMN="Time_years"

###############################################################################
# 时间阈值遍历分析开关（硬编码）
###############################################################################
# YES: 启用时间阈值遍历分析
# NO: 禁用时间阈值遍历分析（默认）
ENABLE_TIME_VARY_ANALYSIS="YES"

# 时间阈值遍历参数（仅在启用时生效）
ANCIENT_START=100000
ANCIENT_END=0
ANCIENT_STEP=500
KERNEL_SIGMA=0.4
TD_SIGMA=0.1

# 为避免在"被删除的工作目录"中运行导致 threadpoolctl/os.getcwd 报错，强制切换到 ROOT。
cd "$ROOT"

mkdir -p "$OUT"

###############################################################################
# 主流程：使用固定 ancient_threshold=50000
###############################################################################
echo "[INFO] ===== 开始主流程分析 (ancient_threshold=50000) ====="

# 1a) 时间深度指标计算
echo "[INFO] 1a) 计算时间深度指标..."
python3 "$PY/1-time_depth.py" \
  --csv_path "$IN_TIME_CSV" \
  --out_csv   "$OUT/time_depth_stats.csv" \
  --group_col "${KEY_COLMN}" \
  --tmrca_col "${TIME_COLMN}" \
  --ancient_threshold 50000 \
  --ratio_quantile 0.01 \
  --kernel_sigma 0.4 \
  --time_depth_sigma_log10 0.1

# 1b) 时间分布指标计算
echo "[INFO] 1b) 计算时间分布指标..."
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

# 1c) 合并时间深度和时间分布指标
echo "[INFO] 1c) 合并时间深度和时间分布指标..."
python3 "$PY/2-2-merge_tmrca_stats.py" \
  --time_depth_csv "$OUT/time_depth_stats.csv" \
  --time_distribution_csv "$OUT/time_distribution_stats.csv" \
  --out_csv "$OUT/Final_tmrca_stats.csv" \
  --group_col "${KEY_COLMN}"

# 2) AMOVA 整理
echo "[INFO] 2) 计算 AMOVA 评分..."
python3 "$PY/3-AMOVA.py" \
  --class_col "${KEY_COLMN}" \
  --variation_type "Source of variation" \
  --variation_value "Percentage of variation" \
  --input  "$IN_AMOVA_CSV" \
  --output "$OUT/Final_AMOVA_scores.csv"

# 3) 单倍群独特性
echo "[INFO] 3) 计算单倍群独特性..."
python3 "$PY/4-Unique.py" \
  --csv "$IN_UNIQ_CSV" \
  --class-col "${KEY_COLMN}" \
  --keyword "$UNIQUE_KEYS" \
  --out-dir "$OUT/"

# 4) 评分判别（四个主要维度评分）
echo "[INFO] 4) 评分判别（四个主要维度评分）..."
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

echo "[INFO] ===== 主流程分析完成 ====="

###############################################################################
# 时间阈值遍历分析（可选）
###############################################################################
if [[ "$ENABLE_TIME_VARY_ANALYSIS" == "YES" ]]; then
  echo ""
  echo "[INFO] ===== 开始时间阈值遍历分析 ====="
  
  TIME_DIR="$OUT/不同时间段"
  mkdir -p "$TIME_DIR"

  # 参数验证
  if (( ANCIENT_STEP <= 0 )); then
    echo "[ERROR] ANCIENT_STEP 必须为正整数，目前为 ${ANCIENT_STEP}" >&2
    exit 1
  fi
  if (( ANCIENT_START < ANCIENT_END )); then
    echo "[ERROR] 需要满足 ANCIENT_START >= ANCIENT_END" >&2
    exit 1
  fi

  COMMON_AMOVA="$TIME_DIR/Final_AMOVA_scores.csv"
  COMMON_UNIQUE="$TIME_DIR/Final_unique_hap.csv"
  AGG_CSV="$TIME_DIR/time_vary_metrics.csv"
  PLOT_PNG="$TIME_DIR/time_vary_metrics.png"

  # 预计算阈值无关的文件：AMOVA 与 Unique Hap
  echo "[INFO] 复制 AMOVA 评分到时间阈值分析目录..."
  cp "$OUT/Final_AMOVA_scores.csv" "$COMMON_AMOVA"

  echo "[INFO] 复制单倍群独特性到时间阈值分析目录..."
  cp "$OUT/Final_unique_hap.csv" "$COMMON_UNIQUE"

  # 主循环：遍历 ancient_threshold
  echo "[INFO] ancient_threshold 从 ${ANCIENT_START} 递减到 ${ANCIENT_END}，步长 ${ANCIENT_STEP}"
  for thr in $(seq "$ANCIENT_START" "-$ANCIENT_STEP" "$ANCIENT_END"); do
    thr_dir="$TIME_DIR/threshold_${thr}"
    mkdir -p "$thr_dir"

    time_depth_csv="$thr_dir/time_depth_stats_thr_${thr}.csv"
    time_dist_csv="$thr_dir/time_distribution_stats_thr_${thr}.csv"
    tmrca_csv="$thr_dir/Final_tmrca_stats_thr_${thr}.csv"
    score_csv="$thr_dir/Final_metrics_scored_thr_${thr}.csv"

    if [[ -f "$score_csv" ]]; then
      echo "[SKIP] 阈值 ${thr} 已存在结果，跳过计算。"
      continue
    fi

    echo "[INFO] >>> 阈值 ${thr} 年：时间深度分析"
    python3 "$PY/1-time_depth.py" \
      --csv_path "$DATA/Time.csv" \
      --out_csv   "$time_depth_csv" \
      --group_col "${KEY_COLMN}" \
      --tmrca_col "${TIME_COLMN}" \
      --ancient_threshold "$thr" \
      --ratio_quantile 0.01 \
      --kernel_sigma "$KERNEL_SIGMA" \
      --time_depth_sigma_log10 "$TD_SIGMA"

    echo "[INFO] >>> 阈值 ${thr} 年：时间分布分析"
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

    echo "[INFO] >>> 阈值 ${thr} 年：合并时间指标"
    python3 "$PY/2-2-merge_tmrca_stats.py" \
      --time_depth_csv "$time_depth_csv" \
      --time_distribution_csv "$time_dist_csv" \
      --out_csv "$tmrca_csv" \
      --group_col "${KEY_COLMN}"

    echo "[INFO] >>> 阈值 ${thr} 年：汇总评分"
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

  # 汇总所有阈值结果
  echo "[INFO] 汇总所有阈值结果到 ${AGG_CSV}"
  python3 "$PY/6-aggregate_time_vary_results.py" "$TIME_DIR" "$AGG_CSV" "${KEY_COLMN}"

  # 可视化：展示不同区域在不同阈值下的多维得分
  echo "[INFO] 绘制可视化结果到 ${PLOT_PNG}"
  python3 "$PY/7-plot_time_vary_metrics.py" "$AGG_CSV" "$PLOT_PNG" "${KEY_COLMN}"

  echo "[INFO] ===== 时间阈值遍历分析完成 ====="
else
  echo ""
  echo "[INFO] 时间阈值遍历分析已禁用。"
  echo "[INFO] 如需启用，请修改脚本中的变量: ENABLE_TIME_VARY_ANALYSIS=\"YES\""
fi

echo ""
echo "[DONE] 全部流程完成。"
