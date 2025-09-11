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
    path1 = tmp_path / "frame1.png"
    cv2.imwrite(str(path1), img1)

    return [path0, path1], img0


def setup(monkeypatch):
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
    return reg_cfg, seg_cfg


def boundary_from(mask):
    boundary = np.zeros_like(mask)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    return boundary


def run_direction(paths, reg_cfg, seg_cfg, direction, tmp_path):
    app_cfg = {"direction": direction, "save_intermediates": True}
    out_dir = tmp_path / f"out_{direction}"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)
    prev_idx = 0 if direction == "first-to-last" else 1
    new_mask = cv2.imread(str(out_dir / "diff" / "new" / f"{prev_idx:04d}_bw_new.png"), cv2.IMREAD_GRAYSCALE)
    lost_mask = cv2.imread(str(out_dir / "diff" / "lost" / f"{prev_idx:04d}_bw_lost.png"), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.imread(str(out_dir / "overlay" / f"{prev_idx:04d}_overlay_mov.png"))
    return new_mask, lost_mask, overlay


def test_new_lost_direction(tmp_path, monkeypatch):
    paths, obj = create_frames(tmp_path)
    reg_cfg, seg_cfg = setup(monkeypatch)
    obj_boundary = boundary_from(obj)

    # first-to-last: object is lost
    new_mask, lost_mask, overlay = run_direction(paths, reg_cfg, seg_cfg, "first-to-last", tmp_path)
    assert np.array_equal(new_mask, np.zeros_like(obj))
    assert np.array_equal(lost_mask, obj)
    red_mask = (overlay == np.array([0, 0, 255], dtype=np.uint8)).all(axis=2).astype(np.uint8) * 255
    assert np.array_equal(red_mask, obj_boundary)
    assert not (overlay == np.array([0, 255, 0], dtype=np.uint8)).all(axis=2).any()

    # last-to-first: object is new
    new_mask, lost_mask, overlay = run_direction(paths, reg_cfg, seg_cfg, "last-to-first", tmp_path)
    assert np.array_equal(new_mask, obj)
    assert np.array_equal(lost_mask, np.zeros_like(obj))
    green_mask = (overlay == np.array([0, 255, 0], dtype=np.uint8)).all(axis=2).astype(np.uint8) * 255
    assert np.array_equal(green_mask, obj_boundary)
    assert not (overlay == np.array([0, 0, 255], dtype=np.uint8)).all(axis=2).any()
