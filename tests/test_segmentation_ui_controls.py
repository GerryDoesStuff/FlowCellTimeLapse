import os
from pathlib import Path
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings
import pyqtgraph as pg

pg.setConfigOptions(useOpenGL=False)

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ui.main_window import MainWindow


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

    # Morphological/removal controls remain enabled
    for w in (win.open_r, win.close_r, win.rm_obj, win.rm_holes):
        assert w.isEnabled()

    win.close()
    app.quit()
