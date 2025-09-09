import os
from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6.QtCore import QSettings
from app.ui.main_window import MainWindow
from app.models.config import save_preset, RegParams, SegParams, AppParams


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
    win.init_radius.setValue(12)
    win.growth_factor.setValue(1.8)
    win.seg_method.setCurrentText("manual")
    win.dir_combo.setCurrentText("first-to-last")
    win.overlay_ref_cb.setChecked(False)
    win.overlay_mov_cb.setChecked(False)
    win.alpha_slider.setValue(75)
    win.close()
    app.processEvents()

    win2 = MainWindow()
    assert win2.max_iters.value() == 321
    assert win2.gauss_sigma.value() == 2.5
    assert win2.clahe_clip.value() == 1.5
    assert win2.clahe_grid.value() == 16
    assert win2.init_radius.value() == 12
    assert win2.growth_factor.value() == 1.8
    assert win2.seg_method.currentText() == "manual"
    assert win2.dir_combo.currentText() == "first-to-last"
    assert not win2.overlay_ref_cb.isChecked()
    assert not win2.overlay_mov_cb.isChecked()
    assert win2.alpha_slider.value() == 75
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
    save_preset(str(preset_file2), RegParams(), SegParams(), AppParams())

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
