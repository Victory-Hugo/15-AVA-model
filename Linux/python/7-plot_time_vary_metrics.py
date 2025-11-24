#!/usr/bin/env python3
"""
Visualize multidimensional scores for each region across different time thresholds
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print("Usage: python 7-plot_time_vary_metrics.py <agg_csv> <output_png> <key_column>")
        sys.exit(1)
    
    agg_csv = Path(sys.argv[1])
    plot_png = Path(sys.argv[2])
    key_column = sys.argv[3]
    
    df = pd.read_csv(agg_csv)
    if df.empty:
        raise SystemExit("[ERROR] Aggregated CSV is empty; cannot plot.")
    
    df = df.sort_values("ancient_threshold")
    metrics = [
        ("time_depth_axis_norm", "Time depth axis (normalized)"),
        ("time_structure_axis_norm", "Time structure complexity (normalized)"),
        ("Diversity_pattern_score_norm", "Genetic diversity score"),
        ("Unique_hap_score_norm", "Unique haplogroup score"),
        ("All_score", "Final overall score"),
    ]
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    
    continents = sorted(df[key_column].dropna().unique())
    colors = plt.cm.get_cmap("tab10", len(continents))
    
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        for c_idx, continent in enumerate(continents):
            subset = df[df[key_column] == continent].sort_values("ancient_threshold")
            ax.plot(
                subset["ancient_threshold"],
                subset[col],
                label=continent if idx == 0 else None,
                color=colors(c_idx),
                marker=".",
                linewidth=1.2,
                markersize=3.5,
            )
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.invert_xaxis()
    
    axes[-1].set_xlabel("ancient_threshold (years)")
    if continents:
        axes[0].legend(title=key_column, loc="best", ncol=2)
    
    fig.tight_layout()
    fig.savefig(plot_png, dpi=300)
    print(f"[OK] Plot saved to {plot_png}")


if __name__ == "__main__":
    main()
