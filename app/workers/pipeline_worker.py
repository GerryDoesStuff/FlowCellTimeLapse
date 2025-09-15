from __future__ import annotations
import logging
from pathlib import Path
from typing import List
import shutil

from PyQt6.QtCore import QObject, pyqtSignal, QThread

from ..core.processing import analyze_sequence
from ..core.io_utils import ensure_dir

ARCHIVE_SUBDIRS = ("registered", "diff", "overlay", "seg")

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
            df.to_csv(self.out_dir / "summary.csv", index=False)

            if self.app_cfg.get("archive_outputs"):
                self._archive_intermediates()

            self.progressed.emit(total, total)
            logger.info("Progressed: %d/%d", total, total)

            self.finished.emit(str(self.out_dir))
            logger.info("Processing finished: %s", self.out_dir)
        except Exception as e:
            logger.exception("Processing failed")
            self.failed.emit(str(e))

    def _archive_intermediates(self):
        for sub in ARCHIVE_SUBDIRS:
            d = self.out_dir / sub
            if d.exists():
                shutil.make_archive(str(d), "zip", root_dir=str(d))
                shutil.rmtree(d, ignore_errors=True)
