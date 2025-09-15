import os
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import pytest

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication

# Ensure app modules can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Use offscreen platform for Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def create_gradient_images(tmp_path, n=2):
    paths = []
    base = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
    for i in range(n):
        p = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(p), base)
        paths.append(p)
    return paths


def test_pipeline_worker_skips_clahe(monkeypatch, tmp_path):
    paths = create_gradient_images(tmp_path, n=2)

    reg_cfg = {
        "model": "translation",
        "max_iters": 10,
        "gauss_blur_sigma": 0,
        "clahe_clip": 2,
        "clahe_grid": 8,
        "use_clahe": False,
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

    def fake_create_clahe(*args, **kwargs):
        raise AssertionError("CLAHE should not be called")

    monkeypatch.setattr(cv2, "createCLAHE", fake_create_clahe)

    import importlib
    from app.core import processing, registration
    import app.workers.pipeline_worker as pw

    importlib.reload(registration)
    importlib.reload(processing)
    importlib.reload(pw)
    PipelineWorker = pw.PipelineWorker

    app = QApplication.instance() or QApplication([])

    worker = PipelineWorker(paths, reg_cfg, seg_cfg, app_cfg, tmp_path / "out")
    captured = {}
    worker.finished.connect(lambda path: captured.setdefault("finished", path))
    worker.failed.connect(lambda msg: captured.setdefault("failed", msg))

    worker.run()

    assert "finished" in captured
    assert "failed" not in captured

    app.quit()
