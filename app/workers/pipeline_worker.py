from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal, QThread

from ..core.processing import analyze_sequence
from ..core.io_utils import ensure_dir

logger = logging.getLogger(__name__)

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
            logger.info("Starting processing for %d paths into %s", len(self.paths), self.out_dir)
            ensure_dir(self.out_dir)

            total = len(self.paths)
            self.progressed.emit(0, total)
            logger.info("Progressed: %d/%d", 0, total)

            df = analyze_sequence(self.paths, self.reg_cfg, self.seg_cfg, self.app_cfg, self.out_dir)

            self.progressed.emit(total, total)
            logger.info("Progressed: %d/%d", total, total)

            self.finished.emit(str(self.out_dir))
            logger.info("Processing finished: %s", self.out_dir)
        except Exception as e:
            logger.exception("Processing failed")
            self.failed.emit(str(e))
