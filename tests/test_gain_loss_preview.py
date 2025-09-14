import os
import sys
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

pg.setConfigOptions(useOpenGL=False)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ui.main_window import MainWindow
from app.core.processing import _detect_green_magenta as real_detect
from app.core.segmentation import segment as real_segment


def test_gain_loss_preview_matches_detection(tmp_path, monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    app = QApplication.instance() or QApplication([])
    win = MainWindow()

    win.seg_method.setCurrentText("manual")
    win.invert.setChecked(False)
    win.skip_outline.setChecked(True)
    win.manual_t.setValue(5)
    win.open_r.setValue(0)
    win.close_r.setValue(0)
    win.rm_obj.setValue(0)
    win.rm_holes.setValue(0)
    win.use_clahe.setChecked(False)
    win.alpha_slider.setValue(60)
    win.gm_sat_slider.setValue(15)

    win._reg_ref = np.array(
        [[10, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        dtype=np.uint8,
    )
    win._reg_warp = np.array(
        [[0, 0, 0],
         [0, 10, 0],
         [0, 0, 0]],
        dtype=np.uint8,
    )

    captured = {}

    def capture_detect(gm_comp, prev_seg, curr_seg, app_cfg, direction):
        g, m = real_detect(gm_comp, prev_seg, curr_seg, app_cfg, direction=direction)
        captured["gm_composite"] = gm_comp
        captured["prev_seg"] = prev_seg
        captured["curr_seg"] = curr_seg
        captured["app_cfg"] = app_cfg
        captured["direction"] = direction
        captured["green_mask"] = g
        captured["magenta_mask"] = m
        return g, m

    monkeypatch.setattr("app.ui.main_window._detect_green_magenta", capture_detect)

    win._diff_gray = np.zeros((3, 3), dtype=np.uint8)
    win._preview_segmentation()
    win._preview_gain_loss()

    alpha = win.alpha_slider.value() / 100.0
    expected_gm = np.zeros((3, 3, 3), dtype=np.uint8)
    expected_gm[..., 1] = (win._reg_ref * (1 - alpha)).astype(np.uint8)
    expected_gm[..., 0] = expected_gm[..., 2] = (win._reg_warp * alpha).astype(np.uint8)

    expected_prev = real_segment(
        win._reg_ref,
        method="manual",
        invert=False,
        skip_outline=True,
        manual_thresh=5,
        adaptive_block=win.adaptive_blk.value(),
        adaptive_C=win.adaptive_C.value(),
        local_block=win.local_blk.value(),
        morph_open_radius=win.open_r.value(),
        morph_close_radius=win.close_r.value(),
        remove_objects_smaller_px=win.rm_obj.value(),
        remove_holes_smaller_px=win.rm_holes.value(),
        use_clahe=win.use_clahe.isChecked(),
    )

    expected_curr = real_segment(
        win._reg_warp,
        method="manual",
        invert=False,
        skip_outline=True,
        manual_thresh=5,
        adaptive_block=win.adaptive_blk.value(),
        adaptive_C=win.adaptive_C.value(),
        local_block=win.local_blk.value(),
        morph_open_radius=win.open_r.value(),
        morph_close_radius=win.close_r.value(),
        remove_objects_smaller_px=win.rm_obj.value(),
        remove_holes_smaller_px=win.rm_holes.value(),
        use_clahe=win.use_clahe.isChecked(),
    )

    app_cfg = {
        "gm_thresh_method": win.gm_thresh_method.currentText(),
        "gm_thresh_percentile": win.gm_thresh_percentile.value(),
        "gm_close_kernel": win.gm_close_k.value(),
        "gm_dilate_kernel": win.gm_dilate_k.value(),
        "gm_saturation": win.gm_sat_slider.value() / 10.0,
    }
    expected_green, expected_magenta = real_detect(
        expected_gm,
        expected_prev,
        expected_curr,
        app_cfg,
        direction=win.dir_combo.currentText(),
    )

    assert np.array_equal(captured["gm_composite"], expected_gm)
    assert np.array_equal(captured["prev_seg"], expected_prev)
    assert np.array_equal(captured["curr_seg"], expected_curr)
    assert captured["app_cfg"] == app_cfg
    assert captured["direction"] == win.dir_combo.currentText()
    assert np.array_equal(captured["green_mask"], expected_green)
    assert np.array_equal(captured["magenta_mask"], expected_magenta)

    win.close()
    app.quit()

def test_gain_loss_controls_remain_enabled_after_param_change(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    app = QApplication.instance() or QApplication([])
    win = MainWindow()

    win.seg_method.setCurrentText("manual")
    win.invert.setChecked(False)
    win.skip_outline.setChecked(True)
    win.manual_t.setValue(5)
    win.open_r.setValue(0)
    win.close_r.setValue(0)
    win.rm_obj.setValue(0)
    win.rm_holes.setValue(0)
    win.use_clahe.setChecked(False)
    win.alpha_slider.setValue(60)
    win.gm_sat_slider.setValue(15)

    win._reg_ref = np.array(
        [[10, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        dtype=np.uint8,
    )
    win._reg_warp = np.array(
        [[0, 0, 0],
         [0, 10, 0],
         [0, 0, 0]],
        dtype=np.uint8,
    )

    win._diff_gray = np.zeros((3, 3), dtype=np.uint8)
    win._preview_segmentation()
    win._preview_gain_loss()

    assert win.gm_section.isEnabled()
    assert win.gm_preview_btn.isEnabled()

    win.gm_close_k.setValue(win.gm_close_k.value() + 1)
    from PyQt6.QtTest import QTest
    QTest.qWait(250)

    assert win.gm_section.isEnabled()
    assert win.gm_preview_btn.isEnabled()

    win.close()
    app.processEvents()
    app.quit()
