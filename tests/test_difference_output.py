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
    seg_dir = out_dir / "seg"
    assert (diff_dir / "raw" / "0001_diff.png").exists()
    assert (diff_dir / "bw" / "0001_bw_diff.png").exists()
    assert (diff_dir / "new" / "0000_bw_new.png").exists()
    assert (diff_dir / "lost" / "0000_bw_lost.png").exists()
    assert (diff_dir / "gain" / "0000_bw_gain.png").exists()
    assert (diff_dir / "loss" / "0000_bw_loss.png").exists()
    assert (diff_dir / "green" / "0000_bw_green.png").exists()
    assert (diff_dir / "magenta" / "0000_bw_magenta.png").exists()
    assert (seg_dir / "mask_0000.png").exists()
    assert (seg_dir / "mask_0000_overlay.png").exists()


def test_difference_output_disabled(tmp_path, monkeypatch):
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
        "use_difference_for_seg": False,
        "save_intermediates": True,
        "save_masks": True,
        "save_diagnostics": False,
    }

    out_dir = tmp_path / "out"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    diff_dir = out_dir / "diff"
    seg_dir = out_dir / "seg"
    assert (diff_dir / "raw" / "0001_diff.png").exists()
    assert (diff_dir / "bw" / "0001_bw_diff.png").exists()
    assert (diff_dir / "green" / "0000_bw_green.png").exists()
    assert (diff_dir / "magenta" / "0000_bw_magenta.png").exists()
    assert not (diff_dir / "new" / "0000_bw_new.png").exists()
    assert not (diff_dir / "lost" / "0000_bw_lost.png").exists()
    assert not (seg_dir / "mask_0000.png").exists()
