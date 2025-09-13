import os
from pathlib import Path
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QSettings
import pyqtgraph as pg

pg.setConfigOptions(useOpenGL=False)

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ui.main_window import MainWindow
from app.models.config import RegParams, SegParams, AppParams


def test_invalid_direction_aborts(tmp_path, monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.paths = [tmp_path / "dummy.png"]

    reg = RegParams()
    seg = SegParams()
    app_params = AppParams(direction="invalid", save_intermediates=False)
    monkeypatch.setattr(win, "_persist_settings", lambda *a, **k: (reg, seg, app_params))

    captured = {}

    def fake_critical(parent, title, text):
        captured["title"] = title
        captured["text"] = text

    monkeypatch.setattr(QMessageBox, "critical", fake_critical)

    win._run_pipeline()

    assert captured["title"] == "Invalid Direction"
    assert "invalid" in captured["text"]
    assert not hasattr(win, "worker")

    win.close()
    app.quit()
