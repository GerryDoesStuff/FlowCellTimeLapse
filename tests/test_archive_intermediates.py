import os
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication

# Ensure app modules can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.workers.pipeline_worker import PipelineWorker


def create_blank_images(tmp_path, n=3):
    paths = []
    for i in range(n):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, 255, -1)
        cv2.line(img, (0, 0), (99, 99), 128, 2)
        cv2.line(img, (99, 0), (0, 99), 128, 2)
        p = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def test_archive_intermediates(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    paths = create_blank_images(tmp_path, n=3)

    reg_cfg = {
        "model": "translation",
        "max_iters": 10,
        "gauss_blur_sigma": 0,
        "clahe_clip": 0,
        "clahe_grid": 8,
        "use_clahe": False,
        "use_masked_ecc": False,
        "method": "ECC",
        "eps": 1e-6,
    }

    seg_cfg = {
        "method": "manual",
        "manual_thresh": 0,
        "invert": True,
        "morph_open_radius": 0,
        "morph_close_radius": 0,
        "remove_objects_smaller_px": 0,
        "remove_holes_smaller_px": 0,
    }

    app_cfg = {
        "direction": "first-to-last",
        "save_diagnostics": True,
        "archive_outputs": True,
    }

    out_dir = tmp_path / "out"
    worker = PipelineWorker(paths, reg_cfg, seg_cfg, app_cfg, out_dir)
    worker.run()

    contents = sorted(p.name for p in out_dir.iterdir())
    assert contents == ["diff.zip", "overlay.zip", "registered.zip", "seg.zip", "summary.csv"]

    app.quit()
