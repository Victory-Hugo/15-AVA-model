#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import threading
import subprocess
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import multiprocessing

import ttkbootstrap as tb
from ttkbootstrap.widgets.scrolled import ScrolledText


def is_pyinstaller_exe():
    """检测是否运行在 PyInstaller 打包的 exe 中"""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def get_python_executable():
    """
    获取 Python 解释器路径。
    - 如果在源码中运行，返回当前解释器
    - 如果在 PyInstaller exe 中运行，返回系统 Python 或 "python"
    """
    if is_pyinstaller_exe():
        # 在 exe 中，使用系统 Python 或直接用 "python" 命令
        return "python"
    else:
        # 在源码中运行，使用当前解释器
        return sys.executable


DEFAULTS = {
    "py_dir": r"../python",
    "key_column": "River",
    "unique_keys": "Yellow River,Yangtze River,Zhujiang River",
    "time_column": "Time_years",
    "in_time_csv": r"Time.csv",
    "in_amova_csv": r"AMOVA.csv",
    "in_uniq_csv": r"public.csv",
    "out_dir": r"/Your output dir/",
    "ancient_threshold": "50000",
    "ratio_quantile": "0.01",
    "kernel_sigma": "0.4",
    "time_depth_sigma_log10": "0.1",
    "gmm_max_components": "5",
    "gmm_min_samples": "10",
    "random_state": "42",
    "skew_method": "auto",
    "variation_type": "Source of variation",
    "variation_value": "Percentage of variation",
    "unique_keyword": "Yellow River,Yangtze River,Zhujiang River",
    "enable_time_vary": "NO",
    "ancient_start": "25000",
    "ancient_end": "0",
    "ancient_step": "500",
    "w_time_depth": "4.0",
    "w_time_structure": "2.0",
    "w_diversity": "1",
    "w_unique": "1",
    "thr_origin": "0.7",
    "thr_mix_low": "0.3",
}


def stream_command(cmd, log_func, description):
    log_func(f"[INFO] {description}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log_func(line.rstrip())
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AVA Model Pipeline")

        self.vars = {}
        for key, value in DEFAULTS.items():
            if key == "enable_time_vary":
                self.vars[key] = tk.BooleanVar(value=str(value).upper() == "YES")
            else:
                self.vars[key] = tk.StringVar(value=value)

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._build_main_tab(notebook)
        self._build_time_depth_tab(notebook)
        self._build_time_distribution_tab(notebook)
        self._build_time_vary_tab(notebook)
        self._build_amova_tab(notebook)
        self._build_unique_tab(notebook)
        self._build_score_tab(notebook)

        control_frame = ttk.Frame(root)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.run_button = ttk.Button(control_frame, text="Run Pipeline", command=self.run_pipeline)
        self.run_button.pack(side=tk.LEFT)

        self.log_box = ScrolledText(root, height=20, state=tk.NORMAL)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _add_labeled_entry(self, parent, text, variable, row, browse=None):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=70)
        entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=4)
        if browse == "file":
            ttk.Button(parent, text="Browse", command=lambda: self._pick_file(variable)).grid(
                row=row, column=2, padx=5, pady=4
            )
        elif browse == "dir":
            ttk.Button(parent, text="Browse", command=lambda: self._pick_dir(variable)).grid(
                row=row, column=2, padx=5, pady=4
            )

    def _pick_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _pick_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _build_main_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Main")
        rows = [
            ("PY scripts folder", "py_dir", "dir"),
            ("KEY_COLUMN", "key_column", None),
            ("UNIQUE_KEYS", "unique_keys", None),
            ("TIME_COLUMN", "time_column", None),
            ("IN_TIME_CSV path", "in_time_csv", "file"),
            ("IN_AMOVA_CSV path", "in_amova_csv", "file"),
            ("IN_UNIQ_CSV path", "in_uniq_csv", "file"),
            ("Output folder (OUT)", "out_dir", "dir"),
        ]
        for idx, (label, key, browse) in enumerate(rows):
            self._add_labeled_entry(frame, label, self.vars[key], idx, browse=browse)

    def _build_time_depth_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Time Depth")
        rows = [
            ("ancient_threshold", "ancient_threshold"),
            ("ratio_quantile", "ratio_quantile"),
            ("kernel_sigma", "kernel_sigma"),
            ("time_depth_sigma_log10", "time_depth_sigma_log10"),
        ]
        for idx, (label, key) in enumerate(rows):
            self._add_labeled_entry(frame, label, self.vars[key], idx)

    def _build_time_distribution_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Time Distribution")
        rows = [
            ("gmm_max_components", "gmm_max_components"),
            ("gmm_min_samples", "gmm_min_samples"),
            ("random_state", "random_state"),
            ("skew_method", "skew_method"),
        ]
        for idx, (label, key) in enumerate(rows):
            self._add_labeled_entry(frame, label, self.vars[key], idx)

    def _build_time_vary_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Time Sweep")
        ttk.Checkbutton(
            frame,
            text="Enable time-vary analysis",
            variable=self.vars["enable_time_vary"],
            onvalue=True,
            offvalue=False,
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        rows = [
            ("ancient_start", "ancient_start"),
            ("ancient_end", "ancient_end"),
            ("ancient_step", "ancient_step"),
        ]
        for idx, (label, key) in enumerate(rows, start=1):
            self._add_labeled_entry(frame, label, self.vars[key], idx)

    def _build_amova_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="AMOVA")
        rows = [
            ("variation_type", "variation_type"),
            ("variation_value", "variation_value"),
        ]
        for idx, (label, key) in enumerate(rows):
            self._add_labeled_entry(frame, label, self.vars[key], idx)

    def _build_unique_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Unique")
        self._add_labeled_entry(frame, "keyword", self.vars["unique_keyword"], 0)
        ttk.Button(
            frame,
            text="Use UNIQUE_KEYS",
            command=lambda: self.vars["unique_keyword"].set(self.vars["unique_keys"].get()),
        ).grid(row=0, column=2, padx=5, pady=4)

    def _build_score_tab(self, notebook):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Score Weights")
        rows = [
            ("w-time-depth", "w_time_depth"),
            ("w-time-structure", "w_time_structure"),
            ("w-diversity", "w_diversity"),
            ("w-unique", "w_unique"),
            ("thr-origin", "thr_origin"),
            ("thr-mix-low", "thr_mix_low"),
        ]
        for idx, (label, key) in enumerate(rows):
            self._add_labeled_entry(frame, label, self.vars[key], idx)

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)
        self.root.update_idletasks()

    def run_pipeline(self):
        self.run_button.config(state=tk.DISABLED)
        self.log_box.delete("1.0", tk.END)
        thread = threading.Thread(target=self._run_pipeline_thread, daemon=True)
        thread.start()

    def _run_pipeline_thread(self):
        try:
            params = {k: v.get() for k, v in self.vars.items()}
            py_dir = Path(params["py_dir"])
            out_dir = Path(params["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd_time_depth = [
                sys.executable,
                str(py_dir / "1-time_depth.py"),
                "--csv_path",
                params["in_time_csv"],
                "--out_csv",
                str(out_dir / "time_depth_stats.csv"),
                "--group_col",
                params["key_column"],
                "--tmrca_col",
                params["time_column"],
                "--ancient_threshold",
                params["ancient_threshold"],
                "--ratio_quantile",
                params["ratio_quantile"],
                "--kernel_sigma",
                params["kernel_sigma"],
                "--time_depth_sigma_log10",
                params["time_depth_sigma_log10"],
            ]

            cmd_time_distribution = [
                sys.executable,
                str(py_dir / "2-1-time_distribution.py"),
                "--csv_path",
                params["in_time_csv"],
                "--out_csv",
                str(out_dir / "time_distribution_stats.csv"),
                "--group_col",
                params["key_column"],
                "--tmrca_col",
                params["time_column"],
                "--ratio_quantile",
                params["ratio_quantile"],
                "--gmm_max_components",
                params["gmm_max_components"],
                "--gmm_min_samples",
                params["gmm_min_samples"],
                "--random_state",
                params["random_state"],
                "--skew_method",
                params["skew_method"],
            ]

            cmd_merge = [
                sys.executable,
                str(py_dir / "2-2-merge_tmrca_stats.py"),
                "--time_depth_csv",
                str(out_dir / "time_depth_stats.csv"),
                "--time_distribution_csv",
                str(out_dir / "time_distribution_stats.csv"),
                "--out_csv",
                str(out_dir / "Final_tmrca_stats.csv"),
                "--group_col",
                params["key_column"],
            ]

            cmd_amova = [
                sys.executable,
                str(py_dir / "3-AMOVA.py"),
                "--class_col",
                params["key_column"],
                "--variation_type",
                params["variation_type"],
                "--variation_value",
                params["variation_value"],
                "--input",
                params["in_amova_csv"],
                "--output",
                str(out_dir / "Final_AMOVA_scores.csv"),
            ]

            cmd_unique = [
                sys.executable,
                str(py_dir / "4-Unique.py"),
                "--csv",
                params["in_uniq_csv"],
                "--class-col",
                params["key_column"],
                "--keyword",
                params["unique_keyword"],
                "--out-dir",
                str(out_dir),
            ]

            cmd_score = [
                sys.executable,
                str(py_dir / "5-score.py"),
                "--tmrca",
                str(out_dir / "Final_tmrca_stats.csv"),
                "--amova",
                str(out_dir / "Final_AMOVA_scores.csv"),
                "--unique",
                str(out_dir / "Final_unique_hap.csv"),
                "--out",
                str(out_dir / "Final_metrics_scored.csv"),
                "--group-col",
                params["key_column"],
                "--w-time-depth",
                params["w_time_depth"],
                "--w-time-structure",
                params["w_time_structure"],
                "--w-diversity",
                params["w_diversity"],
                "--w-unique",
                params["w_unique"],
                "--thr-origin",
                params["thr_origin"],
                "--thr-mix-low",
                params["thr_mix_low"],
            ]

            stream_command(cmd_time_depth, self.log, "1) Time depth metrics")
            stream_command(cmd_time_distribution, self.log, "2) Time distribution metrics")
            stream_command(cmd_merge, self.log, "3) Merge time metrics")
            stream_command(cmd_amova, self.log, "4) AMOVA scores")
            stream_command(cmd_unique, self.log, "5) Unique haplogroups")
            stream_command(cmd_score, self.log, "6) Final scoring")
            if self.vars["enable_time_vary"].get():
                self._run_time_vary_block(params, py_dir, out_dir)
            self.log("[DONE] Pipeline completed")
            messagebox.showinfo("Done", "Pipeline completed successfully.")
        except FileNotFoundError as exc:
            self.log(f"[ERROR] File not found: {exc}")
            messagebox.showerror("Error", f"File not found: {exc}")
        except subprocess.CalledProcessError as exc:
            self.log(f"[ERROR] Command failed with exit code {exc.returncode}")
            messagebox.showerror("Error", f"Command failed. See logs for details.")
        except Exception as exc:  # pragma: no cover - safeguard for unexpected issues
            self.log(f"[ERROR] {exc}")
            messagebox.showerror("Error", str(exc))
        finally:
            self.run_button.config(state=tk.NORMAL)

    def _run_time_vary_block(self, params, py_dir, out_dir):
        try:
            start = int(params["ancient_start"])
            end = int(params["ancient_end"])
            step = int(params["ancient_step"])
        except ValueError:
            raise ValueError("ancient_start/end/step must be integers")

        if step <= 0:
            raise ValueError("ancient_step must be > 0")
        if start < end:
            raise ValueError("ancient_start must be >= ancient_end")

        time_dir = out_dir / "time_vary"
        time_dir.mkdir(parents=True, exist_ok=True)

        common_amova = time_dir / "Final_AMOVA_scores.csv"
        common_unique = time_dir / "Final_unique_hap.csv"
        agg_csv = time_dir / "time_vary_metrics.csv"
        plot_png = time_dir / "time_vary_metrics.png"

        shutil.copy(out_dir / "Final_AMOVA_scores.csv", common_amova)
        shutil.copy(out_dir / "Final_unique_hap.csv", common_unique)

        self.log(f"[INFO] Time sweep from {start} down to {end} step {step}")
        for thr in range(start, end - 1, -step):
            thr_dir = time_dir / f"threshold_{thr}"
            thr_dir.mkdir(parents=True, exist_ok=True)

            time_depth_csv = thr_dir / f"time_depth_stats_thr_{thr}.csv"
            time_dist_csv = thr_dir / f"time_distribution_stats_thr_{thr}.csv"
            tmrca_csv = thr_dir / f"Final_tmrca_stats_thr_{thr}.csv"
            score_csv = thr_dir / f"Final_metrics_scored_thr_{thr}.csv"

            if score_csv.exists():
                self.log(f"[SKIP] threshold {thr} already computed")
                continue

            stream_command(
                [
                    sys.executable,
                    str(py_dir / "1-time_depth.py"),
                    "--csv_path",
                    params["in_time_csv"],
                    "--out_csv",
                    str(time_depth_csv),
                    "--group_col",
                    params["key_column"],
                    "--tmrca_col",
                    params["time_column"],
                    "--ancient_threshold",
                    str(thr),
                    "--ratio_quantile",
                    params["ratio_quantile"],
                    "--kernel_sigma",
                    params["kernel_sigma"],
                    "--time_depth_sigma_log10",
                    params["time_depth_sigma_log10"],
                ],
                self.log,
                f"Time depth @ {thr}",
            )

            stream_command(
                [
                    sys.executable,
                    str(py_dir / "2-1-time_distribution.py"),
                    "--csv_path",
                    params["in_time_csv"],
                    "--out_csv",
                    str(time_dist_csv),
                    "--group_col",
                    params["key_column"],
                    "--tmrca_col",
                    params["time_column"],
                    "--ratio_quantile",
                    params["ratio_quantile"],
                    "--gmm_max_components",
                    params["gmm_max_components"],
                    "--gmm_min_samples",
                    params["gmm_min_samples"],
                    "--random_state",
                    params["random_state"],
                    "--skew_method",
                    params["skew_method"],
                ],
                self.log,
                f"Time distribution @ {thr}",
            )

            stream_command(
                [
                    sys.executable,
                    str(py_dir / "2-2-merge_tmrca_stats.py"),
                    "--time_depth_csv",
                    str(time_depth_csv),
                    "--time_distribution_csv",
                    str(time_dist_csv),
                    "--out_csv",
                    str(tmrca_csv),
                    "--group_col",
                    params["key_column"],
                ],
                self.log,
                f"Merge time metrics @ {thr}",
            )

            stream_command(
                [
                    sys.executable,
                    str(py_dir / "5-score.py"),
                    "--tmrca",
                    str(tmrca_csv),
                    "--amova",
                    str(common_amova),
                    "--unique",
                    str(common_unique),
                    "--out",
                    str(score_csv),
                    "--group-col",
                    params["key_column"],
                    "--w-time-depth",
                    params["w_time_depth"],
                    "--w-time-structure",
                    params["w_time_structure"],
                    "--w-diversity",
                    params["w_diversity"],
                    "--w-unique",
                    params["w_unique"],
                    "--thr-origin",
                    params["thr_origin"],
                    "--thr-mix-low",
                    params["thr_mix_low"],
                ],
                self.log,
                f"Score @ {thr}",
            )

        stream_command(
            [
                sys.executable,
                str(py_dir / "6-aggregate_time_vary_results.py"),
                str(time_dir),
                str(agg_csv),
                params["key_column"],
            ],
            self.log,
            "Aggregate time sweep results",
        )

        stream_command(
            [
                sys.executable,
                str(py_dir / "7-plot_time_vary_metrics.py"),
                str(agg_csv),
                str(plot_png),
                params["key_column"],
            ],
            self.log,
            "Plot time sweep metrics",
        )


def main():
    root = tb.Window(themename="flatly")
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    # 防止在打包为exe后出现无限循环创建窗口的问题
    multiprocessing.freeze_support()
    main()
