import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.processing import analyze_sequence
from app.core import processing


def create_appearance_disappearance_frames(tmp_path):
    # Frame 0: empty
    img0 = np.zeros((32, 32), dtype=np.uint8)
    path0 = tmp_path / "frame0.png"
    cv2.imwrite(str(path0), img0)

    # Frame 1: object appears
    img1 = np.zeros_like(img0)
    cv2.rectangle(img1, (10, 10), (20, 20), 255, -1)
    path1 = tmp_path / "frame1.png"
    cv2.imwrite(str(path1), img1)

    # Frame 2: object disappears
    img2 = np.zeros_like(img0)
    path2 = tmp_path / "frame2.png"
    cv2.imwrite(str(path2), img2)

    return [path0, path1, path2]


def test_overlay_frame_alignment(tmp_path, monkeypatch):
    paths = create_appearance_disappearance_frames(tmp_path)

    def fake_register(ref, mov, model="affine", **kwargs):
        h, w = ref.shape
        mask = np.ones((h, w), dtype=np.uint8)
        return True, np.eye(3, dtype=np.float32), mov, mask

    monkeypatch.setattr(processing, "register_ecc", fake_register)
    # Simple segmentation: foreground where pixel > 0
    monkeypatch.setattr(processing, "segment", lambda img, **kwargs: (img > 0).astype(np.uint8))

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
        "save_diagnostics": True,
    }

    out_dir = tmp_path / "out"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)

    overlay_dir = out_dir / "overlay"
    diff_dir = out_dir / "diff"

    # Appearance between frame0 and frame1 should be attributed to frame0
    assert (overlay_dir / "0000_overlay_mov.png").exists()
    assert (diff_dir / "new" / "0000_bw_new.png").exists()
    bw_new = cv2.imread(str(diff_dir / "gain" / "0000_bw_gain.png"), cv2.IMREAD_GRAYSCALE)
    assert bw_new is not None and np.any(bw_new)

    # Disappearance between frame1 and frame2 should be attributed to frame1
    assert (overlay_dir / "0001_overlay_mov.png").exists()
    assert (diff_dir / "lost" / "0001_bw_lost.png").exists()
    bw_lost = cv2.imread(str(diff_dir / "loss" / "0001_bw_loss.png"), cv2.IMREAD_GRAYSCALE)
    assert bw_lost is not None and np.any(bw_lost)
