import os
from pathlib import Path
import sys
import pytest

pytest.importorskip("PyQt6")
pg = pytest.importorskip("pyqtgraph")
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

pg.setConfigOptions(useOpenGL=False)

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ui.main_window import MainWindow
from app.core.denoise_order import DEFAULT_DENOISE_ORDER


def test_segmentation_method_enables_expected_widgets(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    app = QApplication.instance() or QApplication([])
    win = MainWindow()

    # Manual method
    win.seg_method.setCurrentText("manual")
    assert win.manual_t.isEnabled()
    assert not win.adaptive_blk.isEnabled()
    assert not win.adaptive_C.isEnabled()
    assert not win.local_blk.isEnabled()

    # Adaptive method
    win.seg_method.setCurrentText("adaptive")
    assert not win.manual_t.isEnabled()
    assert win.adaptive_blk.isEnabled()
    assert win.adaptive_C.isEnabled()
    assert not win.local_blk.isEnabled()

    # Local method
    win.seg_method.setCurrentText("local")
    assert not win.manual_t.isEnabled()
    assert not win.adaptive_blk.isEnabled()
    assert not win.adaptive_C.isEnabled()
    assert win.local_blk.isEnabled()

    # Global threshold methods
    for method in ("otsu", "multi_otsu", "li", "yen"):
        win.seg_method.setCurrentText(method)
        assert not win.manual_t.isEnabled()
        assert not win.adaptive_blk.isEnabled()
        assert not win.adaptive_C.isEnabled()
        assert not win.local_blk.isEnabled()

    # Morphological/removal controls remain enabled
    for w in (win.open_r, win.close_r, win.rm_obj, win.rm_holes):
        assert w.isEnabled()

    win.close()
    app.quit()


def test_denoise_subsections_and_params(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    app = QApplication.instance() or QApplication([])
    win = MainWindow()

    expected_attr_order = [
        "gaussian_enabled",
        "median_enabled",
        "bilateral_enabled",
        "nlm_enabled",
        "tv_enabled",
        "anisotropic_enabled",
        "wavelet_enabled",
        "bm3d_enabled",
    ]
    expected_titles = [
        "Gaussian blur",
        "Median filter",
        "Bilateral filter",
        "Fast NLM",
        "Total variation",
        "Anisotropic diffusion",
        "Wavelet",
        "BM3D",
    ]
    assert list(win.denoise_group_boxes.keys()) == expected_attr_order
    assert [box.title() for box in win.denoise_group_boxes.values()] == expected_titles
    assert win._get_denoise_order() == DEFAULT_DENOISE_ORDER

    # Move BM3D to the front and ensure order persists
    new_order = ["bm3d"] + [step for step in DEFAULT_DENOISE_ORDER if step != "bm3d"]
    win._set_denoise_order(new_order, trigger_change=True)
    assert win._get_denoise_order() == new_order
    assert win.seg.denoise_order == new_order
    assert list(win.denoise_group_boxes.keys())[0] == "bm3d_enabled"

    win.denoise_tv_weight.setValue(0.3)
    win.denoise_bm3d_enabled.setChecked(True)
    win.denoise_bm3d_sigma.setValue(25.0)
    win.denoise_bm3d_stage.setCurrentText("soft")

    assert pytest.approx(win.seg.tv_weight, rel=1e-3) == 0.3
    assert win.seg.bm3d_enabled is True
    assert pytest.approx(win.seg.bm3d_sigma, rel=1e-6) == 25.0
    assert win.seg.bm3d_stage == "soft"
    assert win.denoise_bm3d_sigma.isEnabled()
    assert win.denoise_bm3d_stage.isEnabled()

    win.close()
    app.quit()
