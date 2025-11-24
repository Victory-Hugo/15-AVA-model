#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shutil
import threading
import argparse
import importlib.util
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import ttkbootstrap as tb
from ttkbootstrap.widgets.scrolled import ScrolledText


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


_MODULE_CACHE: Dict[Path, ModuleType] = {}


def _make_module_name(path: Path) -> str:
    stem = path.stem.replace("-", "_").replace(".", "_")
    return f"ava_pipeline_{stem}_{abs(hash(path))}"


def _load_run_callable(path: Path) -> Callable[..., Any]:
    resolved = path.resolve()
    if resolved not in _MODULE_CACHE:
        spec = importlib.util.spec_from_file_location(_make_module_name(resolved), resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {resolved}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE_CACHE[resolved] = module
    module = _MODULE_CACHE[resolved]
    run_func = getattr(module, "run", None)
    if not callable(run_func):
        raise AttributeError(f"No callable run() found in {resolved}")
    return run_func


class _LogStream:
    def __init__(self, log_func: Callable[[str], None]) -> None:
        self.log_func = log_func
        self._buffer = ""

    def write(self, text: str) -> None:
        if not text:
            return
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.log_func(line)

    def flush(self) -> None:
        if self._buffer:
            self.log_func(self._buffer)
            self._buffer = ""


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

    def _run_script(self, description: str, script_path: Path, kwargs: Dict[str, Any]):
        self.log(f"[INFO] {description}")
        run_func = _load_run_callable(script_path)
        stream = _LogStream(self.log)
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                return run_func(**kwargs)
        finally:
            stream.flush()

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

            cmd_time_depth = {
                "csv_path": Path(params["in_time_csv"]),
                "out_csv": out_dir / "time_depth_stats.csv",
                "group_col": params["key_column"],
                "tmrca_col": params["time_column"],
                "ancient_threshold": float(params["ancient_threshold"]),
                "ratio_quantile": float(params["ratio_quantile"]),
                "kernel_sigma": float(params["kernel_sigma"]),
                "time_depth_sigma_log10": float(params["time_depth_sigma_log10"]),
                "print_result": True,
            }
            kwargs_time_distribution = {
                "csv_path": Path(params["in_time_csv"]),
                "out_csv": out_dir / "time_distribution_stats.csv",
                "group_col": params["key_column"],
                "tmrca_col": params["time_column"],
                "ratio_quantile": float(params["ratio_quantile"]),
                "gmm_max_components": int(params["gmm_max_components"]),
                "gmm_min_samples": int(params["gmm_min_samples"]),
                "random_state": int(params["random_state"]),
                "skew_method": params["skew_method"],
                "print_result": True,
            }

            kwargs_merge = {
                "time_depth_csv": out_dir / "time_depth_stats.csv",
                "time_distribution_csv": out_dir / "time_distribution_stats.csv",
                "out_csv": out_dir / "Final_tmrca_stats.csv",
                "group_col": params["key_column"],
                "print_result": True,
            }

            kwargs_amova = {
                "class_col": params["key_column"],
                "variation_type": params["variation_type"],
                "variation_value": params["variation_value"],
                "input_path": params["in_amova_csv"],
                "output_path": str(out_dir / "Final_AMOVA_scores.csv"),
                "print_result": True,
            }

            kwargs_unique = {
                "csv_path": params["in_uniq_csv"],
                "class_col": params["key_column"],
                "keyword": params["unique_keyword"],
                "hap_col": "Haplogroup",
                "out_dir": str(out_dir),
                "case_sensitive": False,
                "print_result": True,
            }

            kwargs_score = {
                "tmrca_path": str(out_dir / "Final_tmrca_stats.csv"),
                "amova_path": str(out_dir / "Final_AMOVA_scores.csv"),
                "unique_path": str(out_dir / "Final_unique_hap.csv"),
                "out_path": str(out_dir / "Final_metrics_scored.csv"),
                "group_col": params["key_column"],
                "w_time_depth": float(params["w_time_depth"]),
                "w_time_structure": float(params["w_time_structure"]),
                "w_diversity": float(params["w_diversity"]),
                "w_unique": float(params["w_unique"]),
                "thr_origin": float(params["thr_origin"]),
                "thr_mix_low": float(params["thr_mix_low"]),
                "print_result": True,
            }

            self._run_script("1) Time depth metrics", py_dir / "1-time_depth.py", cmd_time_depth)
            self._run_script("2) Time distribution metrics", py_dir / "2-1-time_distribution.py", kwargs_time_distribution)
            self._run_script("3) Merge time metrics", py_dir / "2-2-merge_tmrca_stats.py", kwargs_merge)
            self._run_script("4) AMOVA scores", py_dir / "3-AMOVA.py", kwargs_amova)
            self._run_script("5) Unique haplogroups", py_dir / "4-Unique.py", kwargs_unique)
            self._run_script("6) Final scoring", py_dir / "5-score.py", kwargs_score)
            if self.vars["enable_time_vary"].get():
                self._run_time_vary_block(params, py_dir, out_dir)
            self.log("[DONE] Pipeline completed")
            messagebox.showinfo("Done", "Pipeline completed successfully.")
        except FileNotFoundError as exc:
            self.log(f"[ERROR] File not found: {exc}")
            messagebox.showerror("Error", f"File not found: {exc}")
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
            td_kwargs = {
                "csv_path": Path(params["in_time_csv"]),
                "out_csv": time_depth_csv,
                "group_col": params["key_column"],
                "tmrca_col": params["time_column"],
                "ancient_threshold": float(thr),
                "ratio_quantile": float(params["ratio_quantile"]),
                "kernel_sigma": float(params["kernel_sigma"]),
                "time_depth_sigma_log10": float(params["time_depth_sigma_log10"]),
                "print_result": True,
            }
            self._run_script(f"Time depth @ {thr}", py_dir / "1-time_depth.py", td_kwargs)

            tdistr_kwargs = {
                "csv_path": Path(params["in_time_csv"]),
                "out_csv": time_dist_csv,
                "group_col": params["key_column"],
                "tmrca_col": params["time_column"],
                "ratio_quantile": float(params["ratio_quantile"]),
                "gmm_max_components": int(params["gmm_max_components"]),
                "gmm_min_samples": int(params["gmm_min_samples"]),
                "random_state": int(params["random_state"]),
                "skew_method": params["skew_method"],
                "print_result": True,
            }
            self._run_script(f"Time distribution @ {thr}", py_dir / "2-1-time_distribution.py", tdistr_kwargs)

            merge_kwargs = {
                "time_depth_csv": time_depth_csv,
                "time_distribution_csv": time_dist_csv,
                "out_csv": tmrca_csv,
                "group_col": params["key_column"],
                "print_result": True,
            }
            self._run_script(f"Merge time metrics @ {thr}", py_dir / "2-2-merge_tmrca_stats.py", merge_kwargs)

            score_kwargs = {
                "tmrca_path": str(tmrca_csv),
                "amova_path": str(common_amova),
                "unique_path": str(common_unique),
                "out_path": str(score_csv),
                "group_col": params["key_column"],
                "w_time_depth": float(params["w_time_depth"]),
                "w_time_structure": float(params["w_time_structure"]),
                "w_diversity": float(params["w_diversity"]),
                "w_unique": float(params["w_unique"]),
                "thr_origin": float(params["thr_origin"]),
                "thr_mix_low": float(params["thr_mix_low"]),
                "print_result": True,
            }
            self._run_script(f"Score @ {thr}", py_dir / "5-score.py", score_kwargs)

        self._run_script(
            "Aggregate time sweep results",
            py_dir / "6-aggregate_time_vary_results.py",
            {
                "base_dir": time_dir,
                "out_csv": agg_csv,
                "group_col": params["key_column"],
                "print_result": True,
            },
        )

        self._run_script(
            "Plot time sweep metrics",
            py_dir / "7-plot_time_vary_metrics.py",
            {
                "agg_csv": agg_csv,
                "out_png": plot_png,
                "group_col": params["key_column"],
                "print_result": True,
            },
        )


def run(argv=None):
    parser = argparse.ArgumentParser(description="AVA Model Pipeline GUI")
    parser.add_argument("--theme", default="flatly", help="ttkbootstrap theme (default: flatly)")
    args = parser.parse_args(argv)
    root = tb.Window(themename=args.theme)
    PipelineGUI(root)
    root.mainloop()


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
