import os
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication


def test_processes_subdirectories_sequentially(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    app = QApplication.instance() or QApplication([])

    from app.ui.main_window import MainWindow

    win = MainWindow()
    win.process_subdirs_cb.setChecked(True)

    sub_a = tmp_path / "a"; sub_a.mkdir()
    sub_b = tmp_path / "b"; sub_b.mkdir()
    img = np.zeros((10, 10), dtype=np.uint8)
    a_img = sub_a / "img.png"; cv2.imwrite(str(a_img), img)
    b_img = sub_b / "img.png"; cv2.imwrite(str(b_img), img)

    win.folder_edit.setText(str(tmp_path))
    win.subdirs = sorted([sub_a, sub_b])
    win.paths = [a_img]

    calls = []

    class DummySignal:
        def __init__(self):
            self._cb = None
        def connect(self, cb):
            self._cb = cb
        def emit(self, *args, **kwargs):
            if self._cb:
                self._cb(*args, **kwargs)

    class DummyThread:
        def __init__(self):
            self.started = DummySignal()
        def start(self):
            self.started.emit()
        def quit(self):
            pass
        def wait(self):
            pass

    class DummyWorker:
        finished = DummySignal()
        failed = DummySignal()
        def __init__(self, paths, reg_cfg, seg_cfg, app_cfg, out_dir):
            calls.append((paths, out_dir, reg_cfg, seg_cfg, app_cfg))
            self.out_dir = out_dir
        def moveToThread(self, thread):
            pass
        def run(self):
            self.finished.emit(str(self.out_dir))

    monkeypatch.setattr("app.ui.main_window.QThread", DummyThread)
    monkeypatch.setattr("app.ui.main_window.PipelineWorker", DummyWorker)

    win._run_pipeline()

    assert len(calls) == 2
    assert Path(calls[0][1]) == sub_a / "_processed_pyqt"
    assert Path(calls[1][1]) == sub_b / "_processed_pyqt"
    assert calls[0][2] == calls[1][2]
    assert calls[0][3] == calls[1][3]
    assert calls[0][4] == calls[1][4]

    win.close()
    app.quit()
