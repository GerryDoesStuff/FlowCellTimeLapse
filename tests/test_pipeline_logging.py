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
