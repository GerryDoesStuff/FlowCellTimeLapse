import os
import sys
from pathlib import Path
import numpy as np
import pytest

pg = pytest.importorskip("pyqtgraph")
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

pg.setConfigOptions(useOpenGL=False)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ui.main_window import MainWindow
from app.core.segmentation import segment as real_segment


def test_segmentation_preview_matches_segment_adaptive(tmp_path, monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))

    app = QApplication.instance() or QApplication([])
    win = MainWindow()

    win.seg_method.setCurrentText("adaptive")
    win.invert.setChecked(False)
    win.skip_outline.setChecked(True)
    win.adaptive_blk.setValue(3)
    win.adaptive_C.setValue(5)
    win.open_r.setValue(0)
    win.close_r.setValue(0)
    win.rm_obj.setValue(0)
    win.rm_holes.setValue(0)
    win.use_clahe.setChecked(False)

    img = np.array(
        [[10, 11, 12],
         [13, 14, 15],
         [16, 17, 18]],
        dtype=np.uint8,
    )

    win._diff_gray = img

    captured = {}

    def capture_segment(gray, **kwargs):
        bw = real_segment(gray, **kwargs)
        captured['bw'] = bw
        return bw

    monkeypatch.setattr("app.ui.main_window.segment", capture_segment)

    win._preview_segmentation()

    expected = real_segment(
        img,
        method="adaptive",
        invert=win.invert.isChecked(),
        skip_outline=win.skip_outline.isChecked(),
        use_diff=True,
        manual_thresh=win.manual_t.value(),
        adaptive_block=win.adaptive_blk.value(),
        adaptive_C=win.adaptive_C.value(),
        local_block=win.local_blk.value(),
        morph_open_radius=win.open_r.value(),
        morph_close_radius=win.close_r.value(),
        remove_objects_smaller_px=win.rm_obj.value(),
        remove_holes_smaller_px=win.rm_holes.value(),
    )

    assert np.array_equal(captured['bw'], expected)

    win.close()
    app.quit()
