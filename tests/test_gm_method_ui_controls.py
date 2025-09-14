import os
import sys
from pathlib import Path
import pytest

pg = pytest.importorskip("pyqtgraph")
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

pg.setConfigOptions(useOpenGL=False)

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ui.main_window import MainWindow


def test_gm_method_enables_expected_widgets(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    app = QApplication.instance() or QApplication([])
    win = MainWindow()

    # Gain/loss section disabled by default; enable to test controls
    win.gm_section.setEnabled(True)
    win._update_gm_controls(win.gm_thresh_method.currentText())

    # Otsu method disables percentile spin box
    win.gm_thresh_method.setCurrentText("otsu")
    assert not win.gm_thresh_percentile.isEnabled()
    for w in (win.gm_close_k, win.gm_dilate_k, win.gm_sat_slider):
        assert w.isEnabled()

    # Percentile method enables percentile spin box
    win.gm_thresh_method.setCurrentText("percentile")
    assert win.gm_thresh_percentile.isEnabled()
    for w in (win.gm_close_k, win.gm_dilate_k, win.gm_sat_slider):
        assert w.isEnabled()

    # Switch back to Otsu
    win.gm_thresh_method.setCurrentText("otsu")
    assert not win.gm_thresh_percentile.isEnabled()

    win.close()
    app.quit()
