import json
import os
import runpy
import shlex
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure UTF-8 aware I/O so bundled scripts can print Unicode bullets/logos even on GBK consoles.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Eagerly import scientific stack so PyInstaller bundles these dependencies.
try:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    from sklearn import mixture as _sklearn_mixture  # noqa: F401
    from sklearn import mixture  # noqa: F401
    import tabulate  # noqa: F401
except Exception:  # pragma: no cover - best effort safeguard
    pass

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QDoubleValidator, QFont, QIntValidator, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def detect_default_root() -> Path:
    """Return the best-guess root folder that holds python/DATA resources."""
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass))
    candidates.append(Path(r"F:\OneDrive\文档（科研）\脚本\Download\15-AVA-model\Win"))
    candidates.append(Path.cwd())

    for candidate in candidates:
        if (candidate / "python").exists():
            return candidate
    return candidates[0]


DEFAULT_ROOT = detect_default_root()
DEFAULT_PY_DIR = DEFAULT_ROOT / "python"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "DATA"
DEFAULT_SAMPLE_DIR = Path.home() / "Desktop"
if getattr(sys, "_MEIPASS", None):
    DEFAULT_OUTPUT_DIR = Path.home() / "AVA_output"
else:
    DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FieldSpec:
    key: str
    label: str
    default: str
    arg: Optional[str] = None
    field_type: str = "text"  # text|int|float
    dialog: Optional[str] = None  # open_file|save_file|directory
    placeholder: str = ""
    tooltip: str = ""
    role: str = "argument"  # argument|script
    visible: bool = True
    readonly: bool = False
    options: Optional[List[str]] = None


class FieldWidget(QWidget):
    def __init__(self, spec: FieldSpec):
        super().__init__()
        self.spec = spec
        self.selector: Optional[QComboBox] = None
        self.line_edit: Optional[QLineEdit] = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        if spec.options:
            self.selector = QComboBox()
            self.selector.addItems(spec.options)
            self.selector.setCurrentText(spec.default)
            if spec.tooltip:
                self.selector.setToolTip(spec.tooltip)
            layout.addWidget(self.selector)
        else:
            self.line_edit = QLineEdit()
            self.line_edit.setText(spec.default)
            if spec.placeholder:
                self.line_edit.setPlaceholderText(spec.placeholder)
            if spec.tooltip:
                self.line_edit.setToolTip(spec.tooltip)
            if spec.readonly:
                self.line_edit.setReadOnly(True)

            if spec.field_type == "int":
                self.line_edit.setValidator(QIntValidator())
            elif spec.field_type == "float":
                validator = QDoubleValidator()
                validator.setNotation(QDoubleValidator.StandardNotation)
                self.line_edit.setValidator(validator)

            layout.addWidget(self.line_edit)

        if spec.dialog:
            browse_btn = QPushButton("Browse")
            browse_btn.setMaximumWidth(90)
            browse_btn.clicked.connect(self.open_dialog)
            layout.addWidget(browse_btn)

    def value(self) -> str:
        if self.selector:
            return self.selector.currentText().strip()
        assert self.line_edit is not None
        return self.line_edit.text().strip()

    def set_value(self, value: str) -> None:
        if self.selector:
            idx = self.selector.findText(value, Qt.MatchExactly)
            if idx >= 0:
                self.selector.setCurrentIndex(idx)
            else:
                self.selector.addItem(value)
                self.selector.setCurrentText(value)
        elif self.line_edit:
            self.line_edit.setText(value)

    def open_dialog(self) -> None:
        if not self.line_edit:
            return
        current = self.value() or str(Path.home())
        if self.spec.dialog == "open_file":
            selected, _ = QFileDialog.getOpenFileName(
                self, f"Select {self.spec.label}", current
            )
        elif self.spec.dialog == "save_file":
            selected, _ = QFileDialog.getSaveFileName(
                self, f"Select {self.spec.label}", current
            )
        elif self.spec.dialog == "directory":
            selected = QFileDialog.getExistingDirectory(
                self, f"Select {self.spec.label}", current
            )
        else:
            selected = ""

        if selected:
            self.set_value(selected)


def run_cli_script(script_path: str, script_args: List[str]) -> int:
    script_file = Path(script_path)
    if not script_file.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        return 1

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        sys.argv = [str(script_file)] + script_args
        os.chdir(script_file.parent)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        runpy.run_path(str(script_file), run_name="__main__")
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 0
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return 1
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        sys.stdout = stdout
        sys.stderr = stderr


def maybe_run_script_mode() -> Optional[int]:
    if "--run-script" not in sys.argv:
        return None

    idx = sys.argv.index("--run-script")
    if idx + 1 >= len(sys.argv):
        print("Missing script path after --run-script", file=sys.stderr)
        return 2
    script_path = sys.argv[idx + 1]
    script_args = sys.argv[idx + 2 :]
    return run_cli_script(script_path, script_args)


def apply_modern_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    window_color = QColor(32, 33, 36)
    base_color = QColor(24, 25, 28)
    text_color = QColor(232, 232, 236)
    disabled_text = QColor(130, 130, 130)
    highlight = QColor(53, 132, 228)

    palette.setColor(QPalette.Window, window_color)
    palette.setColor(QPalette.WindowText, text_color)
    palette.setColor(QPalette.Base, base_color)
    palette.setColor(QPalette.AlternateBase, window_color)
    palette.setColor(QPalette.Text, text_color)
    palette.setColor(QPalette.Button, QColor(45, 45, 48))
    palette.setColor(QPalette.ButtonText, text_color)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)

    app.setPalette(palette)
    app.setFont(QFont("Arial", 10, QFont.Bold))
    app.setStyleSheet(
        """
        QWidget { font-size: 11pt; color: #e8e8ec; font-family: 'Arial'; font-weight: 600; }
        QPushButton {
            background-color: #3578e5;
            border: none;
            border-radius: 4px;
            padding: 6px 14px;
            color: #ffffff;
            font-family: 'Arial';
            font-weight: 600;
        }
        QPushButton:hover { background-color: #4c8df0; }
        QPushButton:pressed { background-color: #2a64c5; }
        QPushButton:disabled { background-color: #3a3a3c; color: #8a8a8d; }
        QLineEdit, QPlainTextEdit {
            border: 1px solid #505054;
            border-radius: 4px;
            padding: 4px;
            background-color: #1f1f22;
            font-family: 'Arial';
            font-weight: 600;
        }
        QTabWidget::pane {
            border: 1px solid #505054;
            border-radius: 4px;
            margin-top: -1px;
        }
        QTabBar::tab {
            background: #2c2c30;
            padding: 6px 12px;
            border: 1px solid #505054;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background: #3b3b40;
        }
        QPlainTextEdit {
            background-color: #121214;
        }
        """
    )


class PipelineWorker(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        stages: List[Tuple[str, List[str]]],
        working_dir: Optional[str] = None,
    ):
        super().__init__()
        self.stages = stages
        self.working_dir = working_dir
        self._stop_requested = False
        self._process: Optional[subprocess.Popen] = None

    def request_stop(self) -> None:
        self._stop_requested = True
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def run(self) -> None:
        try:
            for stage_name, cmd in self.stages:
                if self._stop_requested:
                    self.finished.emit(False, "Pipeline cancelled by user.")
                    return

                self.log.emit(f"\n=== {stage_name} ===")
                self.log.emit(f"Command: {self._format_command(cmd)}")

                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=self.working_dir or None,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                assert self._process.stdout is not None
                for line in self._process.stdout:
                    self.log.emit(line.rstrip())
                    if self._stop_requested:
                        self._process.terminate()
                        self.finished.emit(False, "Pipeline cancelled by user.")
                        return

                return_code = self._process.wait()
                if return_code != 0:
                    self.finished.emit(
                        False,
                        f"{stage_name} failed with exit code {return_code}.",
                    )
                    return

            self.finished.emit(True, "Pipeline completed successfully.")
        except FileNotFoundError as exc:
            self.finished.emit(False, f"File not found: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            self.finished.emit(False, f"Unexpected error: {exc}")

    @staticmethod
    def _format_command(cmd: List[str]) -> str:
        try:
            return shlex.join(cmd)
        except AttributeError:
            return " ".join(f'"{part}"' if " " in part else part for part in cmd)


class PipelineWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AVA Model Pipeline Runner")
        self.resize(960, 720)
        self.stage_specs = self._get_stage_specs()
        self.stage_spec_lookup = {
            stage: {spec.key: spec for spec in specs if spec.role != "script"}
            for stage, specs in self.stage_specs.items()
        }
        self.auto_output_map = {
            ("1) Divergence Time", "out_csv"): "Final_tmrca_stats.csv",
            ("2) AMOVA", "output"): "Final_AMOVA_scores.csv",
            ("3) Unique Haplogroup", "out_dir"): None,
            ("4) Final Score", "tmrca"): "Final_tmrca_stats.csv",
            ("4) Final Score", "amova"): "Final_AMOVA_scores.csv",
            ("4) Final Score", "unique"): "Final_unique_hap.csv",
            ("4) Final Score", "out"): "Final_metrics_scored.csv",
        }
        self.field_widgets: Dict[str, Dict[str, FieldWidget]] = {}
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[PipelineWorker] = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        output_layout = QFormLayout()
        output_layout.setLabelAlignment(Qt.AlignRight)
        self.output_dir_field = FieldWidget(
            FieldSpec(
                key="output_dir",
                label="Output Folder",
                default=str(DEFAULT_OUTPUT_DIR),
                dialog="directory",
                tooltip="All intermediate and final files will be written under this folder.",
            )
        )
        self.output_dir_field.line_edit.textChanged.connect(
            lambda _: self._sync_output_paths()
        )
        output_layout.addRow("Output Folder:", self.output_dir_field)
        main_layout.addLayout(output_layout)

        # Stage tabs
        self.tabs = QTabWidget()
        self._init_stage_tabs()
        main_layout.addWidget(self.tabs)
        self._sync_output_paths()

        # Action buttons
        button_row = QHBoxLayout()
        self.run_button = QPushButton("Run Pipeline")
        self.run_button.clicked.connect(self.start_pipeline)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_pipeline)

        self.save_button = QPushButton("Save Config")
        self.save_button.clicked.connect(self.save_config)

        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.load_config)

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)

        for btn in (
            self.run_button,
            self.cancel_button,
            self.save_button,
            self.load_button,
            self.clear_log_button,
        ):
            button_row.addWidget(btn)

        button_row.addStretch()
        main_layout.addLayout(button_row)

        # Log output
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log_output.setMaximumBlockCount(10000)
        self.log_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.log_output)

        # Status label
        self.status_label = QLabel("Idle")
        main_layout.addWidget(self.status_label)

    def _init_stage_tabs(self) -> None:
        for stage_name, specs in self.stage_specs.items():
            tab = QWidget()
            layout = QFormLayout(tab)
            layout.setLabelAlignment(Qt.AlignRight)
            widgets: Dict[str, FieldWidget] = {}

            for spec in specs:
                if spec.role == "script":
                    continue
                field = FieldWidget(spec)
                if spec.visible:
                    layout.addRow(f"{spec.label}:", field)
                widgets[spec.key] = field

            self.field_widgets[stage_name] = widgets
            self.tabs.addTab(tab, stage_name)

    def _get_base_output_dir(self) -> Path:
        base = self.output_dir_field.value().strip() or str(DEFAULT_OUTPUT_DIR)
        path = Path(base).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _sync_output_paths(self) -> None:
        try:
            base_path = self._get_base_output_dir()
        except OSError:
            return

        for (stage_name, key), relative in self.auto_output_map.items():
            widget = self.field_widgets.get(stage_name, {}).get(key)
            if not widget:
                continue
            path = base_path if relative is None else base_path / relative
            widget.set_value(str(path))

    @staticmethod
    def _get_stage_specs() -> Dict[str, List[FieldSpec]]:
        return {
            "1) Divergence Time": [
                FieldSpec(
                    key="script",
                    label="Script Path",
                    default=str(DEFAULT_PY_DIR / "1-divergence_time_shape.py"),
                    dialog="open_file",
                    role="script",
                    visible=False,
                ),
                FieldSpec(
                    key="csv_path",
                    label="Input CSV",
                    default=str(DEFAULT_SAMPLE_DIR / "1-ID_Time_Class.csv"),
                    arg="--csv_path",
                    dialog="open_file",
                ),
                FieldSpec(
                    key="out_csv",
                    label="Output CSV",
                    default=str(DEFAULT_OUTPUT_DIR / "Final_tmrca_stats.csv"),
                    arg="--out_csv",
                    dialog=None,
                    visible=False,
                    readonly=True,
                ),
                FieldSpec(
                    key="group_col",
                    label="Group Column",
                    default="Continent",
                    arg="--group_col",
                ),
                FieldSpec(
                    key="tmrca_col",
                    label="TMRCA Column",
                    default="Time_years",
                    arg="--tmrca_col",
                ),
                FieldSpec(
                    key="ancient_threshold",
                    label="Ancient Threshold",
                    default="100000",
                    arg="--ancient_threshold",
                    field_type="int",
                ),
                FieldSpec(
                    key="ratio_quantile",
                    label="Ratio Quantile",
                    default="0.01",
                    arg="--ratio_quantile",
                    field_type="float",
                ),
                FieldSpec(
                    key="gmm_max_components",
                    label="GMM Max Components",
                    default="5",
                    arg="--gmm_max_components",
                    field_type="int",
                ),
                FieldSpec(
                    key="gmm_min_samples",
                    label="GMM Min Samples",
                    default="10",
                    arg="--gmm_min_samples",
                    field_type="int",
                ),
                FieldSpec(
                    key="random_state",
                    label="Random State",
                    default="42",
                    arg="--random_state",
                    field_type="int",
                ),
                FieldSpec(
                    key="skew_method",
                    label="Skew Method",
                    default="auto",
                    arg="--skew_method",
                    options=["auto", "moment", "quantile"],
                ),
            ],
            "2) AMOVA": [
                FieldSpec(
                    key="script",
                    label="Script Path",
                    default=str(DEFAULT_PY_DIR / "2-AMOVA.py"),
                    dialog="open_file",
                    role="script",
                    visible=False,
                ),
                FieldSpec(
                    key="input",
                    label="Input CSV",
                    default=str(DEFAULT_SAMPLE_DIR / "2-AMOVA.csv"),
                    arg="--input",
                    dialog="open_file",
                ),
                FieldSpec(
                    key="class_col",
                    label="Class Column",
                    default="Continent",
                    arg="--class_col",
                ),
                FieldSpec(
                    key="variation_type",
                    label="Variation Type Column",
                    default="Source of variation",
                    arg="--variation_type",
                ),
                FieldSpec(
                    key="variation_value",
                    label="Variation Value Column",
                    default="Percentage of variation",
                    arg="--variation_value",
                ),
                FieldSpec(
                    key="output",
                    label="Output CSV",
                    default=str(DEFAULT_OUTPUT_DIR / "Final_AMOVA_scores.csv"),
                    arg="--output",
                    dialog=None,
                    visible=False,
                    readonly=True,
                ),
            ],
            "3) Unique Haplogroup": [
                FieldSpec(
                    key="script",
                    label="Script Path",
                    default=str(DEFAULT_PY_DIR / "3-unique_haplogroup.py"),
                    dialog="open_file",
                    role="script",
                    visible=False,
                ),
                FieldSpec(
                    key="csv",
                    label="Input CSV",
                    default=str(DEFAULT_SAMPLE_DIR / "3-public.csv"),
                    arg="--csv",
                    dialog="open_file",
                ),
                FieldSpec(
                    key="class_col",
                    label="Class Column",
                    default="Continent",
                    arg="--class-col",
                ),
                FieldSpec(
                    key="keyword",
                    label="Keyword List",
                    default="Africa,Central_Asia,Southeast_Asia",
                    arg="--keyword",
                    tooltip="Comma-separated list used to detect unique haplogroups.",
                ),
                FieldSpec(
                    key="out_dir",
                    label="Output Directory",
                    default=str(DEFAULT_OUTPUT_DIR),
                    arg="--out-dir",
                    dialog=None,
                    visible=False,
                    readonly=True,
                ),
            ],
            "4) Final Score": [
                FieldSpec(
                    key="script",
                    label="Script Path",
                    default=str(DEFAULT_PY_DIR / "4-score.py"),
                    dialog="open_file",
                    role="script",
                    visible=False,
                ),
                FieldSpec(
                    key="tmrca",
                    label="TMRCA CSV",
                    default=str(DEFAULT_OUTPUT_DIR / "Final_tmrca_stats.csv"),
                    arg="--tmrca",
                    dialog=None,
                    visible=False,
                    readonly=True,
                ),
                FieldSpec(
                    key="amova",
                    label="AMOVA CSV",
                    default=str(DEFAULT_OUTPUT_DIR / "Final_AMOVA_scores.csv"),
                    arg="--amova",
                    dialog=None,
                    visible=False,
                    readonly=True,
                ),
                FieldSpec(
                    key="unique",
                    label="Unique Haplogroup CSV",
                    default=str(DEFAULT_OUTPUT_DIR / "Final_unique_hap.csv"),
                    arg="--unique",
                    dialog=None,
                    visible=False,
                    readonly=True,
                ),
                FieldSpec(
                    key="out",
                    label="Output CSV",
                    default=str(DEFAULT_OUTPUT_DIR / "Final_metrics_scored.csv"),
                    arg="--out",
                    dialog=None,
                    visible=False,
                    readonly=True,
                ),
                FieldSpec(
                    key="w_max",
                    label="Weight Max",
                    default="2",
                    arg="--w-max",
                    field_type="float",
                ),
                FieldSpec(
                    key="w_ancient",
                    label="Weight Ancient",
                    default="2",
                    arg="--w-ancient",
                    field_type="float",
                ),
                FieldSpec(
                    key="w_std",
                    label="Weight Std",
                    default="1",
                    arg="--w-std",
                    field_type="float",
                ),
                FieldSpec(
                    key="w_range",
                    label="Weight Range",
                    default="1",
                    arg="--w-range",
                    field_type="float",
                ),
                FieldSpec(
                    key="w_skew",
                    label="Weight Skew",
                    default="1",
                    arg="--w-skew",
                    field_type="float",
                ),
                FieldSpec(
                    key="w_peaks",
                    label="Weight Peaks",
                    default="1",
                    arg="--w-peaks",
                    field_type="float",
                ),
                FieldSpec(
                    key="w_diversity",
                    label="Weight Diversity",
                    default="1.5",
                    arg="--w-diversity",
                    field_type="float",
                ),
                FieldSpec(
                    key="w_unique",
                    label="Weight Unique",
                    default="1.5",
                    arg="--w-unique",
                    field_type="float",
                ),
                FieldSpec(
                    key="thr_origin",
                    label="Threshold Origin",
                    default="0.7",
                    arg="--thr-origin",
                    field_type="float",
                ),
                FieldSpec(
                    key="thr_mix_low",
                    label="Threshold Mix Low",
                    default="0.3",
                    arg="--thr-mix-low",
                    field_type="float",
                ),
            ],
        }

    def start_pipeline(self) -> None:
        try:
            base_output_dir = self._get_base_output_dir()
        except OSError as exc:
            QMessageBox.critical(self, "Invalid output folder", str(exc))
            return

        self._sync_output_paths()
        try:
            stages = self._build_stage_commands()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc))
            return

        self.log_output.appendPlainText("Starting pipeline...\n")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Running...")

        self.worker = PipelineWorker(stages)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._pipeline_finished)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def cancel_pipeline(self) -> None:
        if self.worker:
            self.worker.request_stop()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling...")

    def _pipeline_finished(self, success: bool, message: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None
        self.worker_thread = None
        self.status_label.setText("Completed" if success else "Failed")
        self._append_log(message)
        if success:
            print("Pipeline finished successfully.")
        else:
            print(f"Pipeline failed: {message}")

    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def _build_stage_commands(self) -> List[Tuple[str, List[str]]]:
        commands: List[Tuple[str, List[str]]] = []
        for stage_name, widgets in self.field_widgets.items():
            script_path = self._require_path(
                self._get_script_path(stage_name), f"{stage_name} script"
            )
            cmd = [sys.executable, "--run-script", script_path]
            for spec_key, field in widgets.items():
                spec = self._find_spec(stage_name, spec_key)
                if not spec.arg:
                    continue
                value = field.value()
                if not value:
                    raise ValueError(f"{spec.label} in '{stage_name}' cannot be empty.")
                if spec.field_type == "int":
                    value = str(int(float(value)))
                elif spec.field_type == "float":
                    value = str(float(value))
                cmd.extend([spec.arg, value])
            commands.append((stage_name, cmd))
        return commands

    def _get_script_path(self, stage_name: str) -> str:
        for spec in self.stage_specs.get(stage_name, []):
            if spec.role == "script":
                return str(Path(spec.default).resolve())
        raise KeyError(f"No script defined for stage '{stage_name}'.")

    def _find_spec(self, stage_name: str, key: str) -> FieldSpec:
        spec = self.stage_spec_lookup.get(stage_name, {}).get(key)
        if spec:
            return spec
        raise KeyError(f"Unknown field {key} in {stage_name}")

    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", str(Path.home() / "ava_pipeline.json"), "JSON (*.json)"
        )
        if not path:
            return

        data = {
            "output_dir": self.output_dir_field.value(),
            "stages": {
                stage: {key: widget.value() for key, widget in widgets.items()}
                for stage, widgets in self.field_widgets.items()
            },
        }

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", f"Configuration saved to {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {exc}")

    def load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load configuration", str(Path.home()), "JSON (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Error", f"Failed to load configuration: {exc}")
            return

        self.output_dir_field.set_value(
            data.get("output_dir", str(DEFAULT_OUTPUT_DIR))
        )

        stages: Dict[str, Dict[str, str]] = data.get("stages", {})
        for stage_name, widgets in self.field_widgets.items():
            stage_values = stages.get(stage_name, {})
            for key, widget in widgets.items():
                if key in stage_values:
                    widget.set_value(stage_values[key])

        QMessageBox.information(self, "Success", f"Configuration loaded from {path}")

    def clear_log(self) -> None:
        self.log_output.clear()

    @staticmethod
    def _require_path(path_str: str, label: str) -> str:
        if not path_str:
            raise ValueError(f"{label} cannot be empty.")
        path = Path(path_str)
        if not path.exists():
            raise ValueError(f"{label} does not exist: {path_str}")
        return str(path)


def main() -> None:
    script_exit = maybe_run_script_mode()
    if script_exit is not None:
        sys.exit(script_exit)

    app = QApplication(sys.argv)
    apply_modern_theme(app)
    window = PipelineWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
