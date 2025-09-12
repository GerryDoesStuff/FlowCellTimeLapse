import numpy as np
import sys
from pathlib import Path
import cv2

# Ensure the application package is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.segmentation import segment


def test_segment_invokes_clahe(monkeypatch):
    calls = {}

    class FakeClahe:
        def apply(self, img):
            calls["apply"] = True
            return img

    def fake_create(*args, **kwargs):
        calls["create"] = True
        return FakeClahe()

    monkeypatch.setattr(cv2, "createCLAHE", fake_create)

    img = np.full((10, 10), 128, dtype=np.uint8)
    segment(img, use_clahe=True)

    assert "create" in calls
    assert "apply" in calls

