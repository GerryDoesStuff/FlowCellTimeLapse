import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence
from app.core import processing


def create_simple_frames(tmp_path):
    img0 = np.zeros((32, 32), dtype=np.uint8)
    cv2.rectangle(img0, (5, 5), (15, 15), 255, -1)
    path0 = tmp_path / "frame0.png"
    cv2.imwrite(str(path0), img0)

    img1 = np.zeros_like(img0)
    cv2.rectangle(img1, (8, 5), (18, 15), 255, -1)
    path1 = tmp_path / "frame1.png"
    cv2.imwrite(str(path1), img1)

    return [path0, path1]


def test_difference_output(tmp_path, monkeypatch):
    paths = create_simple_frames(tmp_path)

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        mask = np.ones((h, w), dtype=np.uint8)
        return True, np.eye(3, dtype=np.float32), mov, mask

    monkeypatch.setattr(processing, "register_ecc", fake_register)
    monkeypatch.setattr(processing, "segment", lambda img, **kwargs: np.ones_like(img, dtype=np.uint8))

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
        "use_difference_for_seg": True,
        "save_intermediates": True,
        "save_masks": True,
    }

    out_dir = tmp_path / "out"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    diff_dir = out_dir / "diff"
    assert (diff_dir / "0001_diff.png").exists()
    assert (diff_dir / "0000_bw_new.png").exists()
    assert (diff_dir / "0000_bw_lost.png").exists()

    reg0 = cv2.imread(str(out_dir / "mask_0000_registered.png"), cv2.IMREAD_GRAYSCALE)
    reg1 = cv2.imread(str(out_dir / "mask_0001_registered.png"), cv2.IMREAD_GRAYSCALE)
    diff1 = cv2.imread(str(out_dir / "mask_0001_difference.png"), cv2.IMREAD_GRAYSCALE)
    assert reg0 is not None and reg0.shape == (32, 32)
    assert reg1 is not None and reg1.shape == (32, 32)
    assert diff1 is not None and diff1.shape == (32, 32)
