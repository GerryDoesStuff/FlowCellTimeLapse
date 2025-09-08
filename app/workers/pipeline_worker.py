from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from pathlib import Path
from typing import List, Dict, Any
from ..core.processing import analyze_sequence
from ..core.io_utils import ensure_dir

class PipelineWorker(QObject):
    progressed = pyqtSignal(int, int)  # current, total (reserved for future granular progress)
    finished = pyqtSignal(str)         # output dir
    failed = pyqtSignal(str)

    def __init__(self, paths: list[Path], reg_cfg: dict, seg_cfg: dict, app_cfg: dict, out_dir: Path):
        super().__init__()
        self.paths = paths
        self.reg_cfg = reg_cfg
        self.seg_cfg = seg_cfg
        self.app_cfg = app_cfg
        self.out_dir = out_dir

    def run(self):
        try:
            ensure_dir(self.out_dir)
            df = analyze_sequence(self.paths, self.reg_cfg, self.seg_cfg, self.app_cfg, self.out_dir)
            self.finished.emit(str(self.out_dir))
        except Exception as e:
            self.failed.emit(str(e))
