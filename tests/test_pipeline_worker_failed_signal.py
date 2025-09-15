import os
import numpy as np
import cv2
from pathlib import Path
import sys
import pytest
import os

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.workers.pipeline_worker import PipelineWorker
from app.core import processing


def create_blank_images(tmp_path, n=2):
    paths = []
    for i in range(n):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
        paths.append(tmp_path / f"img_{i}.png")
    return paths


def test_pipeline_worker_emits_failed_on_low_overlap(tmp_path, monkeypatch):
    paths = create_blank_images(tmp_path, n=2)

    reg_cfg = {
        "model": "translation",
        "max_iters": 10,
        "gauss_blur_sigma": 0,
        "clahe_clip": 0,
        "clahe_grid": 8,
        "use_masked_ecc": False,
        "method": "ECC",
        "eps": 1e-6,
        "growth_factor": 1.0,
        "initial_radius": 0,
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

    app_cfg = {"direction": "first-to-last", "save_diagnostics": False}

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        valid = np.zeros((h, w), dtype=np.uint8)
        valid[0, 0] = 1
        return True, np.eye(3, dtype=np.float32), mov.copy(), valid

    monkeypatch.setattr(processing, "register_ecc", fake_register)

    app = QApplication.instance() or QApplication([])

    worker = PipelineWorker(paths, reg_cfg, seg_cfg, app_cfg, tmp_path / "out")

    captured = {}

    worker.failed.connect(lambda msg: captured.setdefault("failed", msg))
    worker.finished.connect(lambda path: captured.setdefault("finished", path))

    worker.run()

    assert "failed" in captured
    assert "overlap area" in captured["failed"].lower()
    assert "finished" not in captured

    app.quit()
