import os

import sys
from pathlib import Path

import numpy as np
import cv2
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.mark.parametrize("method", ["otsu", "multi_otsu", "li", "yen"])
def test_run_pipeline_passes_segmentation_params(tmp_path, monkeypatch, method):
    """Ensure seg_cfg includes all preview params and is passed unchanged."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    app = QApplication.instance() or QApplication([])

    from app.ui.main_window import MainWindow

    win = MainWindow()
    win.seg_method.setCurrentText(method)

    # Create a dummy image path so _run_pipeline has something to process
    img = np.zeros((10, 10), dtype=np.uint8)
    img_path = tmp_path / "img.png"
    cv2.imwrite(str(img_path), img)
    win.paths = [img_path]

    # Toggle skip_outline to verify it propagates
    win.skip_outline.setChecked(True)

    captured = {}

    class DummySignal:
        def connect(self, *args, **kwargs):
            pass

    class DummyThread:
        def __init__(self):
            self.started = DummySignal()

        def start(self):
            pass

        def quit(self):
            pass

        def wait(self):
            pass

    class DummyWorker:
        finished = DummySignal()
        failed = DummySignal()

        def __init__(self, paths, reg_cfg, seg_cfg, app_cfg, out_dir):
            captured["seg_cfg"] = seg_cfg

        def moveToThread(self, thread):
            pass

        def run(self):
            pass

    monkeypatch.setattr("app.ui.main_window.QThread", DummyThread)
    monkeypatch.setattr("app.ui.main_window.PipelineWorker", DummyWorker)

    # Run the pipeline; our dummy worker will capture seg_cfg
    win._run_pipeline()

    # Build expected seg_cfg from current segmentation settings
    _, seg, _ = win._persist_settings()
    expected = dict(
        method=seg.method,
        invert=seg.invert,
        skip_outline=seg.skip_outline,
        manual_thresh=seg.manual_thresh,
        adaptive_block=seg.adaptive_block,
        adaptive_C=seg.adaptive_C,
        local_block=seg.local_block,
        morph_open_radius=seg.morph_open_radius,
        morph_close_radius=seg.morph_close_radius,
        remove_objects_smaller_px=seg.remove_objects_smaller_px,
        remove_holes_smaller_px=seg.remove_holes_smaller_px,
    )

    assert captured["seg_cfg"] == expected

    win.close()
    app.quit()

