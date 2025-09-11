import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence
from app.core import processing

def create_frames(tmp_path):
    img0 = np.zeros((32, 32), dtype=np.uint8)
    cv2.rectangle(img0, (5, 5), (15, 15), 255, -1)
    path0 = tmp_path / "frame0.png"
    cv2.imwrite(str(path0), img0)

    img1 = np.zeros_like(img0)
    cv2.rectangle(img1, (10, 5), (20, 15), 255, -1)
    path1 = tmp_path / "frame1.png"
    cv2.imwrite(str(path1), img1)

    return [path0, path1]

def test_overlay_contains_new_and_lost_colors(tmp_path, monkeypatch):
    paths = create_frames(tmp_path)

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        mask = np.ones((h, w), dtype=np.uint8)
        return True, np.eye(3, dtype=np.float32), mov, mask

    monkeypatch.setattr(processing, "register_ecc", fake_register)
    monkeypatch.setattr(processing, "segment", lambda img, **kwargs: (img > 127).astype(np.uint8))

    reg_cfg = {
        "initial_radius": 0,
        "gauss_blur_sigma": 0,
        "clahe_clip": 0,
        "clahe_grid": 8,
        "use_masked_ecc": False,
    }
    seg_cfg = {}
    app_cfg = {
        "direction": "first-to-last",
        "save_intermediates": True,
        "overlay_new_color": (0, 255, 0),
        "overlay_lost_color": (0, 0, 255),
    }

    out_dir = tmp_path / "out"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    overlay_img = cv2.imread(str(out_dir / "overlay" / "0001_overlay_mov.png"))
    assert overlay_img is not None
    green = np.array([0, 255, 0], dtype=np.uint8)
    red = np.array([0, 0, 255], dtype=np.uint8)
    assert (overlay_img == green).all(axis=2).any()
    assert (overlay_img == red).all(axis=2).any()
