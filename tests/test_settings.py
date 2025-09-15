import os
import pytest

pytest.importorskip("PyQt6")
pg = pytest.importorskip("pyqtgraph")
from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6.QtCore import QSettings

pg.setConfigOptions(useOpenGL=False)

from app.ui.main_window import MainWindow
from app.models.config import save_preset, load_preset, RegParams, SegParams, AppParams


def test_settings_persist(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    s = QSettings("YeastLab", "FlowcellPyQt")
    s.clear()
    s.sync()

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.max_iters.setValue(321)
    win.gauss_sigma.setValue(2.5)
    win.clahe_clip.setValue(1.5)
    win.clahe_grid.setValue(16)
    win.use_clahe.setChecked(False)
    win.init_radius.setValue(12)
    win.growth_factor.setValue(1.8)
    win.reg_method.setCurrentText("ORB")
    win.reg_model.setCurrentText("translation")
    win.eps.setValue(5e-5)
    win.use_masked.setChecked(False)
    win.orb_features.setValue(123)
    win.match_ratio.setValue(0.55)
    win.min_keypoints.setValue(5)
    win.min_matches.setValue(6)
    win.use_ecc_fallback.setChecked(False)
    win.seg_method.setCurrentText("manual")
    win.skip_outline.setChecked(True)
    win.dir_combo.setCurrentText("first-to-last")
    win.diff_method.setCurrentText("lab")
    win.overlay_ref_cb.setChecked(False)
    win.overlay_mov_cb.setChecked(False)
    win.alpha_slider.setValue(75)
    win.overlay_mode_combo.setCurrentText("grayscale")
    win.norm_cb.setChecked(True)
    win.scale_min.setValue(5)
    win.scale_max.setValue(100)
    win.bg_sub_cb.setChecked(True)
    win.save_intermediates.setChecked(True)
    win.archive_intermediates.setChecked(True)
    win.save_diag_checkbox.setChecked(False)
    win.gm_sat_slider.setValue(15)
    win.close()
    app.processEvents()

    win2 = MainWindow()
    assert win2.max_iters.value() == 321
    assert win2.gauss_sigma.value() == 2.5
    assert win2.clahe_clip.value() == 1.5
    assert win2.clahe_grid.value() == 16
    assert not win2.use_clahe.isChecked()
    assert win2.init_radius.value() == 12
    assert win2.growth_factor.value() == 1.8
    assert win2.reg_method.currentText() == "ORB"
    assert win2.reg_model.currentText() == "translation"
    assert win2.eps.value() == 5e-05
    assert not win2.use_masked.isChecked()
    assert win2.orb_features.value() == 123
    assert win2.match_ratio.value() == 0.55
    assert win2.min_keypoints.value() == 5
    assert win2.min_matches.value() == 6
    assert not win2.use_ecc_fallback.isChecked()
    assert win2.seg_method.currentText() == "manual"
    assert win2.skip_outline.isChecked()
    assert win2.dir_combo.currentText() == "first-to-last"
    assert win2.diff_method.currentText() == "lab"
    assert not win2.overlay_ref_cb.isChecked()
    assert not win2.overlay_mov_cb.isChecked()
    assert win2.alpha_slider.value() == 75
    assert win2.overlay_mode_combo.currentText() == "grayscale"
    assert win2.norm_cb.isChecked()
    assert win2.scale_min.value() == 5
    assert win2.scale_max.value() == 100
    assert win2.bg_sub_cb.isChecked()
    assert win2.save_intermediates.isChecked()
    assert win2.archive_intermediates.isChecked()
    assert not win2.save_diag_checkbox.isChecked()
    assert win2.gm_sat_slider.value() == 15
    win2.close()
    app.quit()


def test_presets_path_persist(tmp_path, monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    s = QSettings("YeastLab", "FlowcellPyQt")
    s.clear(); s.sync()

    preset_dir1 = tmp_path / "presets1"
    preset_dir1.mkdir()
    preset_file1 = preset_dir1 / "preset1.json"

    def fake_save(parent, caption, dir, filter):
        return str(preset_file1), "JSON (*.json)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_save)

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win._save_preset()
    assert win.app.presets_path == str(preset_dir1)
    win.close()
    app.processEvents()

    # Persisted in settings
    win2 = MainWindow()
    assert win2.app.presets_path == str(preset_dir1)

    # Prepare second preset to load
    preset_dir2 = tmp_path / "presets2"
    preset_dir2.mkdir()
    preset_file2 = preset_dir2 / "preset2.json"
    save_preset(str(preset_file2), RegParams(), SegParams(), AppParams(save_intermediates=False))

    def fake_open(parent, caption, dir, filter):
        assert dir == str(preset_dir1)
        return str(preset_file2), "JSON (*.json)"

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_open)
    win2._load_preset()
    assert win2.app.presets_path == str(preset_dir2)
    win2.close()

    # Ensure path persisted
    win3 = MainWindow()
    assert win3.app.presets_path == str(preset_dir2)
    win3.close()
    app.quit()


def test_preset_gm_params(tmp_path):
    preset = tmp_path / "preset.json"
    save_preset(
        str(preset),
        RegParams(),
        SegParams(),
        AppParams(gm_opacity=67, gm_saturation=1.5),
    )
    _, _, app = load_preset(str(preset))
    assert app.gm_opacity == 67
    assert app.gm_saturation == 1.5
