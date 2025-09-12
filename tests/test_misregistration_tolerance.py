import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure application package importable when tests run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core import processing
from app.core.processing import analyze_sequence


def create_shifted_frames(tmp_path, dx, dy):
    base = np.zeros((32, 32), dtype=np.uint8)
    cv2.rectangle(base, (5, 5), (15, 15), 255, -1)
    img0 = base
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img1 = cv2.warpAffine(base, M, (32, 32), flags=cv2.INTER_NEAREST, borderValue=0)
    p0 = tmp_path / "frame0.png"
    p1 = tmp_path / "frame1.png"
    cv2.imwrite(str(p0), img0)
    cv2.imwrite(str(p1), img1)
    return [p0, p1]


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


@pytest.mark.parametrize(
    "dx,dy,dilate",
    [
        (1, 0, 3),
        (-1, 0, 3),
        (0, 1, 3),
        (0, -1, 3),
        (2, 0, 5),
        (-2, 0, 5),
        (0, 2, 5),
        (0, -2, 5),
    ],
)
def test_misregistration_tolerance(tmp_path, monkeypatch, dx, dy, dilate):
    paths = create_shifted_frames(tmp_path, dx, dy)
    reg_cfg, seg_cfg = setup(monkeypatch)
    app_cfg = {
        "direction": "first-to-last",
        "save_intermediates": True,
        "class_dilate_kernel": dilate,
        "component_min_overlap": 0.5,
    }
    out_dir = tmp_path / "out"
    analyze_sequence(paths, reg_cfg, seg_cfg, app_cfg, out_dir)
    new_mask = cv2.imread(str(out_dir / "diff" / "new" / "0000_bw_new.png"), cv2.IMREAD_GRAYSCALE)
    lost_mask = cv2.imread(str(out_dir / "diff" / "lost" / "0000_bw_lost.png"), cv2.IMREAD_GRAYSCALE)
    assert new_mask is not None and not np.any(new_mask)
    assert lost_mask is not None and not np.any(lost_mask)
