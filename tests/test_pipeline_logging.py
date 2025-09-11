import os
from pathlib import Path
import sys
import logging
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings, QThread
import pyqtgraph as pg

pg.setConfigOptions(useOpenGL=False)

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ui.main_window import MainWindow
from app.models.config import RegParams, SegParams, AppParams
from app.core.processing import analyze_sequence
import numpy as np
import cv2


@pytest.mark.parametrize("direction", ["first-to-last", "last-to-first"])
def test_pipeline_logs_direction(tmp_path, monkeypatch, caplog, direction):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.paths = [tmp_path / "dummy.png"]
    win.folder_edit.setText(str(tmp_path))

    reg = RegParams()
    seg = SegParams()
    app_params = AppParams(direction=direction)
    monkeypatch.setattr(win, "_persist_settings", lambda *a, **k: (reg, seg, app_params))
    monkeypatch.setattr(QThread, "start", lambda self: None)

    with caplog.at_level(logging.INFO):
        win._run_pipeline()

    assert f"direction={direction}" in caplog.text

    win.close()
    app.quit()


def test_save_masks(tmp_path, monkeypatch):
    paths = []
    img0 = np.zeros((30, 30), dtype=np.uint8)
    cv2.rectangle(img0, (5, 5), (15, 15), 255, -1)
    p0 = tmp_path / "img0.png"; cv2.imwrite(str(p0), img0); paths.append(p0)
    img1 = np.zeros((30, 30), dtype=np.uint8)
    cv2.rectangle(img1, (10, 10), (20, 20), 255, -1)
    p1 = tmp_path / "img1.png"; cv2.imwrite(str(p1), img1); paths.append(p1)

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
        "manual_thresh": 1,
        "invert": False,
        "morph_open_radius": 0,
        "morph_close_radius": 0,
        "remove_objects_smaller_px": 0,
        "remove_holes_smaller_px": 0,
    }

    app_cfg = {"direction": "first-to-last", "save_intermediates": False, "save_masks": True}

    from app.core import processing

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        valid = np.ones((h, w), dtype=np.uint8)
        return True, np.eye(3, dtype=np.float32), mov.copy(), valid

    monkeypatch.setattr(processing, "register_ecc", fake_register)

    out_dir = tmp_path / "out"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    mask0_path = out_dir / "mask_0000_registered.png"
    mask1_path = out_dir / "mask_0001_registered.png"
    assert mask0_path.exists()
    assert mask1_path.exists()
    mask0 = cv2.imread(str(mask0_path), cv2.IMREAD_GRAYSCALE)
    mask1 = cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE)
    assert mask0.shape == img0.shape
    assert mask1.shape == img1.shape
