import os
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

pg = pytest.importorskip("pyqtgraph")
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

pg.setConfigOptions(useOpenGL=False)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ui.main_window import MainWindow


def _write(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def test_diff_overlay_toggle(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    app = QApplication.instance() or QApplication([])
    win = MainWindow()

    diff_dir = tmp_path / "diff"
    raw = np.zeros((3, 3), dtype=np.uint8)
    raw[1, 1] = 100
    green = np.zeros((3, 3), dtype=np.uint8)
    green[0, 0] = 255
    magenta = np.zeros((3, 3), dtype=np.uint8)
    magenta[2, 2] = 255
    _write(diff_dir / "raw" / "0000_diff.png", raw)
    _write(diff_dir / "green" / "0000_bw_green.png", green)
    _write(diff_dir / "magenta" / "0000_bw_magenta.png", magenta)

    win.show_diff_overlay_cb.setChecked(True)
    win._update_preview(diff_dir, 0)

    expected = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    for mask, color in ((green, (0, 255, 0)), (magenta, (255, 0, 255))):
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(expected, contours, -1, color, 1)
    expected_disp = cv2.cvtColor(expected, cv2.COLOR_BGR2RGB).transpose(1, 0, 2)
    assert np.array_equal(win.view.imageItem.image.astype(np.uint8), expected_disp)

    win.show_diff_overlay_cb.setChecked(False)
    win._update_preview(diff_dir, 0)
    expected_disp2 = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB).transpose(1, 0, 2)
    assert np.array_equal(win.view.imageItem.image.astype(np.uint8), expected_disp2)

    win.close()
    app.quit()

